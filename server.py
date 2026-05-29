import asyncio
import json
import logging
import socket
import threading
import time
import warnings
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any, NoReturn

# LiteLLM's internal ModelResponse/StreamingChoices schemas don't always match
# what the runtime objects actually carry; Pydantic emits cosmetic
# UserWarnings during serialization. Output is still correct.
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\..*")

import litellm
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse

# Silence LiteLLM's "Provider List: https://…" debug print that accompanies
# provider-detection errors.
litellm.suppress_debug_info = True
litellm.drop_params=True

logger = logging.getLogger(__name__)

def _pick_unused_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _get_default_ipv4_address() -> str:
    """Mirror Agent Lightning's advertised-host behavior for 0.0.0.0 binds."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


class Server:
    """Route Claude Code Anthropic messages requests through LiteLLM's Python SDK."""

    stopped: bool

    def __init__(self, 
                 sonnet_model: str = "claude-sonnet-4-6", 
                 haiku_model: str = "claude-haiku-4-5-20251001"
                ) -> None:
        self.stopped = True
        self._app: FastAPI | None = None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._base_url: str | None = None
        self._auth_token: str = "dummy"
        self._model_aliases: dict[str, str] = {}
        self._litellm_params: dict[str, Any] = {}
        self.sonnet_model=sonnet_model
        self.haiku_model=haiku_model

    def start(self, **kwargs: Any) -> tuple[str, str]:
        """Start a minimal Anthropic-compatible LiteLLM SDK server.

        Returns:
            base_url, auth_token suitable for ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN.
        """
        self.stop()

        model = kwargs.pop("model")
        host = kwargs.pop("host", "0.0.0.0")
        port = int(kwargs.pop("port", 0) or _pick_unused_port())
        access_host = kwargs.pop("access_host", None)
        auth_token = kwargs.pop("auth_token", "dummy")
        log_level = kwargs.pop("log_level", "warning")
        startup_timeout = float(kwargs.pop("startup_timeout", 60.0))
        num_retries = kwargs.pop("num_retries", 0)

        sonnet_model = kwargs.pop("sonnet_model", kwargs.pop("model_high", model))
        haiku_model = kwargs.pop("haiku_model", kwargs.pop("model_low", model))
        sonnet_name = kwargs.pop("sonnet_name", kwargs.pop("frontend_model_high", self.sonnet_model))
        haiku_name = kwargs.pop("haiku_name", kwargs.pop("frontend_model_low", self.haiku_model))
        model_aliases = kwargs.pop("model_aliases", None)

        api_base = kwargs.pop("api_base", kwargs.pop("base_url", None))
        litellm_params = dict(kwargs)
        if api_base is not None:
            litellm_params["api_base"] = api_base
        litellm_params.setdefault("num_retries", num_retries)

        self._auth_token = auth_token
        self._model_aliases = model_aliases or {
            sonnet_name: sonnet_model,
            haiku_name: haiku_model,
        }
        self._litellm_params = {key: value for key, value in litellm_params.items() if value is not None}
        self._app = self._build_app()

        config = uvicorn.Config(self._app, host=host, port=port, log_level=log_level, access_log=False)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        self._wait_until_ready(host, port, startup_timeout)

        advertised_host = access_host or (_get_default_ipv4_address() if host in ("0.0.0.0", "::") else host)
        self._base_url = f"http://{advertised_host}:{port}"
        self.stopped = False
        logger.info("Started Claude Code LiteLLM SDK server at %s", self._base_url)
        return self._base_url, self._auth_token

    def start_from_api_key(self, model: str, base_url: str, api_key: str, **kwargs: Any) -> tuple[str, str]:
        """Start from a provider model, base URL, and API key."""
        return self.start(model=model, base_url=base_url, api_key=api_key, **kwargs)

    def start_from_azure_openai(self, model: str, **kwargs: Any) -> tuple[str, str]:
        """Start from the CloudGPT Azure OpenAI endpoint used by the existing runner."""
        raise NotImplementedError("For azure openai indentity token login, you should prepare the script to get azure openai token provider yourself")
        #from <your_script> import get_openai_token_provider

        token_provider = get_openai_token_provider()
        return self.start(
            model=model,
            azure_ad_token_provider=token_provider,
            **kwargs,
        )

    def stop(self) -> None:
        if self.stopped:
            return

        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)

        self._server = None
        self._thread = None
        self._app = None
        self._base_url = None
        self.stopped = True

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.post("/v1/messages", response_model=None)
        async def messages(request: Request) -> JSONResponse | StreamingResponse:
            body = await request.json()
            model = self._resolve_model(body.get("model"))
            payload = {**body, "model": model, **self._litellm_params}

            try:
                response = await litellm.anthropic.messages.acreate(**payload)
            except Exception as exc:
                return self._anthropic_error_response(exc)

            if body.get("stream", False):
                return StreamingResponse(
                    self._sse_chunks(response),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
                )

            return JSONResponse(content=self._to_jsonable(response))

        @app.post("/v1/messages/count_tokens")
        async def count_tokens(request: Request) -> dict[str, int]:
            body = await request.json()
            # Count against the alias the client requested (Anthropic-side), not the
            # upstream model — Claude Code budgets context based on the name it sent.
            alias_model = str(body.get("model") or "")
            if alias_model not in self._model_aliases:
                self._resolve_model(body.get("model"))  # raises a clean 400
            tokens = await asyncio.to_thread(
                litellm.token_counter,
                model=alias_model,
                messages=body.get("messages", []),
            )
            return {"input_tokens": int(tokens)}

        return app

    def _resolve_model(self, requested_model: Any) -> str:
        if requested_model in self._model_aliases:
            return self._model_aliases[str(requested_model)]

        raise HTTPException(
            status_code=400,
            detail={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        f"Invalid model name passed in model={requested_model}. "
                        f"Available models: {', '.join(self._model_aliases)}"
                    ),
                },
            },
        )

    async def _sse_chunks(self, response: Any) -> AsyncIterator[bytes | str]:
        if not hasattr(response, "__aiter__"):
            yield f"event: message\ndata: {json.dumps(self._to_jsonable(response))}\n\n"
            return

        try:
            async for chunk in response:
                if isinstance(chunk, (bytes, str)):
                    yield chunk
                    continue

                data = self._to_jsonable(chunk)
                if self._is_empty_stream_choice(data):
                    continue

                event_type = data.get("type", "message") if isinstance(data, dict) else "message"
                yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            raise
        except IndexError as exc:
            if not self._is_litellm_empty_choices_error(exc):
                raise
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        except Exception as exc:
            # Emit an Anthropic-style error event followed by a terminal message_stop
            # so Claude Code closes the stream cleanly instead of hanging.
            logger.exception("Streaming response failed mid-iteration")
            error_payload = self._anthropic_error_payload(exc)
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        finally:
            await self._close_stream(response)

    @staticmethod
    def _is_empty_stream_choice(data: Any) -> bool:
        return isinstance(data, dict) and data.get("choices") == []

    @staticmethod
    def _is_litellm_empty_choices_error(exc: IndexError) -> bool:
        return str(exc) == "list index out of range"

    async def _close_stream(self, response: Any) -> None:
        with suppress(Exception):
            aclose = getattr(response, "aclose", None)
            if aclose is not None:
                await aclose()
                return

            completion_stream = getattr(response, "completion_stream", None)
            if completion_stream is not None:
                await self._close_stream(completion_stream)

    def _wait_until_ready(self, host: str, port: int, startup_timeout: float) -> None:
        connect_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            if self._thread is not None and not self._thread.is_alive():
                raise RuntimeError(
                    f"uvicorn thread died during startup on {host}:{port}. "
                    "Check logs for the underlying exception (port in use, app build failure, etc.)."
                )
            if self._server and self._server.started:
                if self._healthcheck(connect_host, port):
                    return
            time.sleep(0.1)
        raise RuntimeError(f"Timed out waiting for server to start on {host}:{port}")

    @staticmethod
    def _healthcheck(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.2) as sock:
                request = f"GET /health HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
                sock.sendall(request.encode("ascii"))
                return b" 200 " in sock.recv(128)
        except OSError:
            return False

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        return jsonable_encoder(value)

    @staticmethod
    def _anthropic_error_payload(exc: Exception) -> dict[str, Any]:
        message = getattr(exc, "message", None) or str(exc)
        error_type = getattr(exc, "type", None) or type(exc).__name__
        return {"type": "error", "error": {"type": str(error_type), "message": str(message)}}

    @classmethod
    def _anthropic_error_response(cls, exc: Exception) -> JSONResponse:
        status_code = int(getattr(exc, "status_code", 500) or 500)
        return JSONResponse(status_code=status_code, content=cls._anthropic_error_payload(exc))

    @classmethod
    def _raise_http_error(cls, exc: Exception) -> NoReturn:
        status_code = int(getattr(exc, "status_code", 500) or 500)
        raise HTTPException(status_code=status_code, detail=cls._anthropic_error_payload(exc))

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
