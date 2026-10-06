"""FastAPI gateway with OpenAI-compatible endpoints."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from .config import Config
from .keystore import KeyStore

logger = logging.getLogger(__name__)

# `store` and `stream_options` are OpenAI-only/streaming-usage fields the
# backend chat-completions endpoint rejects with 400s; `tools`/`tool_choice`/
# `parallel_tool_calls` are supported by the fleet's vLLM build
# (--enable-auto-tool-choice --tool-call-parser poolside_v1) and MUST reach
# the backend, or vLLM never activates its tool-call parser and the model's
# tool calls fall through as raw unparsed text in the response content.
_UNSUPPORTED = {"store", "stream_options"}


@dataclass
class BackendEntry:
    """A live, registered backend for one configured backend pool."""

    url: str  # "http://node:port"
    model_name: str
    job_id: str


class Gateway:
    """OpenAI-compatible gateway proxy, routing across multiple backend pools."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.app = FastAPI(title="tequila-mule", version="0.1.0")
        self.backends: Dict[str, List[BackendEntry]] = {}  # keyed by backend pool name
        self.backend_lock = asyncio.Lock()
        self.client = httpx.AsyncClient(timeout=300.0)  # 5min for long completions
        # Round-robin counters for load balancing across replicas
        self._rr_counters: Dict[str, int] = {}
        # Valid backend_names for registration (primary + secondary model names)
        self._valid_backend_names: set[str] = set()
        for backend in config.backends:
            self._valid_backend_names.add(backend.name)
            for sec in backend.secondary_models:
                self._valid_backend_names.add(sec.backend_name)

        # Static routing table: canonical backend name, model name, and every
        # alias all resolve to the owning backend pool's name (or secondary model's
        # backend_name for secondary models).
        self._route_table: Dict[str, str] = self._build_route_table(config)

        # Load keystore
        keystore_path = Path(config.paths.api_keys_file).expanduser()
        self.keystore = KeyStore(keystore_path)

        # Register routes
        self.app.post("/v1/chat/completions")(self._chat_completions)
        self.app.post("/v1/completions")(self._completions)
        self.app.post("/v1/embeddings")(self._embeddings)
        self.app.get("/v1/models")(self._models)
        self.app.post("/internal/register")(self._register_backend)
        self.app.get("/health")(self._health)

    @staticmethod
    def _build_route_table(config: Config) -> Dict[str, str]:
        table: Dict[str, str] = {}
        for backend in config.backends:
            # Primary model routing
            table[backend.name] = backend.name
            table[backend.model.name] = backend.name
            for alias in backend.aliases:
                table[alias] = backend.name
            # Secondary model routing
            for sec in backend.secondary_models:
                table[sec.backend_name] = sec.backend_name
                table[sec.model_name] = sec.backend_name
                for alias in sec.aliases:
                    table[alias] = sec.backend_name
        return table

    def _next_entry(self, backend_name: str) -> Optional[BackendEntry]:
        """Round-robin pick among a pool's registered instances."""
        entries = self.backends.get(backend_name)
        if not entries:
            return None
        counter = self._rr_counters.get(backend_name, 0)
        self._rr_counters[backend_name] = counter + 1
        return entries[counter % len(entries)]

    def get_backend(self, backend_name: str) -> Optional[str]:
        """Return a URL for the pool, round-robin across its instances."""
        entry = self._next_entry(backend_name)
        return entry.url if entry else None

    def get_backend_job_id(self, backend_name: str) -> Optional[str]:
        """Return the job_id of the backend currently registered for a pool, if any.

        Returns the job_id from the first registered entry (all entries in a pool
        share the same job_id since they're launched by the same Slurm job).
        """
        entries = self.backends.get(backend_name, [])
        return entries[0].job_id if entries else None

    def get_all_backend_urls(self, backend_name: str) -> list[str]:
        """Return all registered URLs for a pool (for health monitoring)."""
        entries = self.backends.get(backend_name, [])
        return [e.url for e in entries]

    def get_backend_urls_by_job(self, backend_name: str, job_id: str) -> list[str]:
        """Get all URLs for a pool matching a specific job_id."""
        entries = self.backends.get(backend_name, [])
        return [e.url for e in entries if e.job_id == job_id]

    async def clear_backend(self, backend_name: str) -> None:
        """Remove the registered backend(s) for a pool."""
        async with self.backend_lock:
            entries = self.backends.pop(backend_name, [])
            if entries:
                for entry in entries:
                    logger.info(f"[{backend_name}] Cleared backend registration: {entry.url}")
            self._rr_counters.pop(backend_name, None)

    async def set_backend(
        self, backend_name: str, backend_url: str, model_name: str = "", job_id: str = ""
    ) -> None:
        """Register a live instance for a pool.

        Instances from the same job accumulate (replicas). The first
        registration from a different job_id replaces every older entry —
        that is the rotation flip to the new job.
        """
        async with self.backend_lock:
            entries = [e for e in self.backends.get(backend_name, []) if e.job_id == job_id]
            if any(e.url == backend_url for e in entries):
                return
            entries.append(BackendEntry(url=backend_url, model_name=model_name, job_id=job_id))
            self.backends[backend_name] = entries
            logger.info(f"[{backend_name}] Registered backend: {backend_url} (job: {job_id})")

    async def _check_auth(self, request: Request) -> Optional[str]:
        """Check API key if auth enabled. Returns email if authenticated."""
        if not self.keystore.has_keys():
            return None  # Auth disabled (no keys configured)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid Authorization header",
            )

        token = auth_header[7:]  # Strip "Bearer "
        email = self.keystore.verify_key(token)
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        return email

    async def _proxy_request(
        self, request: Request, endpoint: str, stream: bool = False
    ) -> Response:
        """Resolve the target backend by model/alias and proxy the request."""
        await self._check_auth(request)

        try:
            body_json = json.loads(await request.body())
        except json.JSONDecodeError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body")

        requested_model = body_json.get("model")
        backend_name = self._route_table.get(requested_model) if requested_model else None
        if backend_name is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model: {requested_model!r}",
            )

        entry = self._next_entry(backend_name)
        if entry is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Backend for model {requested_model!r} not ready",
                headers={"Retry-After": "60"},
            )

        url = f"{entry.url}{endpoint}"
        # Rewrite the model field to the backend's actual served model name so
        # clients can use a canonical name or alias. Also strip OpenAI fields
        # the backend chat-completions endpoint has no use for (see
        # _UNSUPPORTED) to avoid 400 validation errors.
        body_json["model"] = entry.model_name
        for key in _UNSUPPORTED:
            body_json.pop(key, None)
        body = json.dumps(body_json).encode()

        try:
            if stream:
                # Stream response from backend, normalizing SSE chunks to
                # match the OpenAI spec: the final chunk must have an empty
                # delta ({}) with finish_reason set, not delta={"content":""}.
                # If the backend returns a non-2xx status, forward it as JSON
                # (headers not yet sent at that point so we can still do this).
                async def stream_response():
                    async with self.client.stream(
                        "POST",
                        url,
                        content=body,
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        if resp.status_code >= 400:
                            error_body = await resp.aread()
                            logger.error(f"Backend stream error {resp.status_code}: {error_body[:200]}")
                            # Emit a terminal SSE error so the client gets a
                            # meaningful failure instead of an empty stream.
                            err = json.dumps({"error": {"message": error_body.decode(errors="replace"), "type": "backend_error", "code": resp.status_code}})
                            yield b"data: " + err.encode() + b"\n\n"
                            yield b"data: [DONE]\n\n"
                            return
                        async for line in resp.aiter_lines():
                            if not line:
                                continue  # skip blank separator lines
                            if not line.startswith("data: "):
                                continue  # skip other fields (event:, id:, etc.)
                            payload = line[6:]
                            if payload == "[DONE]":
                                yield b"data: [DONE]\n\n"
                                continue
                            try:
                                chunk = json.loads(payload)
                                for choice in chunk.get("choices", []):
                                    delta = choice.get("delta", {})
                                    # vLLM 0.4.x bundles content="" with
                                    # finish_reason in the last chunk; OpenAI
                                    # spec wants delta={} for that event.
                                    if choice.get("finish_reason") and delta.get("content") == "":
                                        choice["delta"] = {}
                                chunk.pop("usage", None)
                                yield b"data: " + json.dumps(chunk).encode() + b"\n\n"
                            except (json.JSONDecodeError, KeyError):
                                yield b"data: " + payload.encode() + b"\n\n"

                return StreamingResponse(
                    stream_response(),
                    media_type="text/event-stream",
                )
            else:
                resp = await self.client.post(
                    url,
                    content=body,
                    headers={"Content-Type": "application/json"},
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type="application/json",
                )

        except httpx.RequestError as e:
            logger.error(f"Backend request failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Backend unreachable",
            )

    async def _chat_completions(self, request: Request) -> Response:
        """OpenAI chat completions endpoint."""
        body = await request.json()
        stream = body.get("stream", False)
        logger.info(f"chat/completions: stream={stream} model={body.get('model')} messages={len(body.get('messages', []))}")
        return await self._proxy_request(request, "/v1/chat/completions", stream=stream)

    async def _completions(self, request: Request) -> Response:
        """OpenAI completions endpoint."""
        body = await request.json()
        stream = body.get("stream", False)
        return await self._proxy_request(request, "/v1/completions", stream=stream)

    async def _embeddings(self, request: Request) -> Response:
        """OpenAI embeddings endpoint (served by pooling-model backends)."""
        return await self._proxy_request(request, "/v1/embeddings")

    async def _models(self, request: Request) -> Response:
        """OpenAI models endpoint. Lists canonical model names and aliases
        for every currently-registered (live) backend."""
        await self._check_auth(request)

        created = int(time.time())

        def model_obj(model_id: str, root: str) -> dict:
            # Full OpenAI-compatible model object. Some harnesses validate
            # strictly against this schema, so we populate every standard field
            # (created/owned_by/root/parent/permission) rather than the bare
            # id/object pair. `root` links an alias back to the served model.
            return {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "tequila-mule",
                "root": root,
                "parent": None if model_id == root else root,
                "permission": [
                    {
                        "id": f"modelperm-{model_id}",
                        "object": "model_permission",
                        "created": created,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    }
                ],
            }

        served = []  # (backend_name, model id, aliases) for every pool and secondary
        for backend in self.config.backends:
            served.append((backend.name, backend.model.name, backend.aliases))
            served += [(s.backend_name, s.model_name, s.aliases) for s in backend.secondary_models]

        data = []
        for backend_name, model_name, aliases in served:
            if not self.backends.get(backend_name):
                continue  # no live instance yet; don't advertise it
            data.append(model_obj(model_name, model_name))
            data += [model_obj(alias, model_name) for alias in aliases]

        if not data:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No backend available",
            )

        return Response(
            content=json.dumps({"object": "list", "data": data}),
            media_type="application/json",
        )

    async def _register_backend(self, request: Request) -> dict:
        """Internal endpoint for backends to register themselves."""
        data = await request.json()
        job_id = data["job_id"]
        node = data["node"]
        port = data["port"]
        backend_name = data.get("backend_name")
        model_name = data.get("model_name", "")

        if backend_name not in self._valid_backend_names:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown backend_name: {backend_name!r}",
            )

        backend_url = f"http://{node}:{port}"
        await self.set_backend(backend_name, backend_url, model_name=model_name, job_id=job_id)

        return {"status": "registered", "backend": backend_url, "backend_name": backend_name}

    async def _health(self) -> dict:
        """Gateway health check."""
        backends_info = {}
        for name, entries in self.backends.items():
            backends_info[name] = [e.url for e in entries]
        return {
            "status": "healthy",
            "backends": backends_info,
        }

    async def shutdown(self) -> None:
        """Shutdown gateway."""
        await self.client.aclose()
