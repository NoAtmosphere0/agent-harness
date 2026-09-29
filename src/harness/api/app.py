"""FastAPI application: lifespan, error mapping and routes.

``uvicorn harness.api.app:app`` (or ``harness serve``) builds everything from
env vars. Tests call ``create_app(container)`` with their own container.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from harness import __version__
from harness.api.routes import router
from harness.config import get_settings
from harness.domain.errors import (
    ApprovalNotFoundError,
    ApprovalNotPendingError,
    InvalidConfigOverridesError,
    RunAlreadyTerminalError,
    RunNotFoundError,
    RunNotWaitingError,
    ServiceError,
)
from harness.observability.logging import configure_logging, get_logger
from harness.wiring import Container, build_container

WAIT_TIMEOUT_S = 60.0

_HTTP_STATUS: dict[type[ServiceError], int] = {
    RunNotFoundError: status.HTTP_404_NOT_FOUND,
    ApprovalNotFoundError: status.HTTP_404_NOT_FOUND,
    ApprovalNotPendingError: status.HTTP_409_CONFLICT,
    RunNotWaitingError: status.HTTP_409_CONFLICT,
    RunAlreadyTerminalError: status.HTTP_409_CONFLICT,
    InvalidConfigOverridesError: status.HTTP_422_UNPROCESSABLE_CONTENT,
}

_log = get_logger(component="api")


def create_app(container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = container is None
        if container is None:
            settings = get_settings()
            configure_logging(settings.log_level, settings.log_format)
            app.state.container = await build_container(settings)
        try:
            yield
        finally:
            c: Container = app.state.container
            # §12: runs still being advanced are failed, not left RUNNING.
            if owned:
                await c.close()
            else:
                await c.runs.shutdown()

    app = FastAPI(title="Agent Harness", version=__version__, lifespan=lifespan)
    app.state.container = container
    app.state.wait_timeout_s = WAIT_TIMEOUT_S
    app.include_router(router)
    app.add_exception_handler(ServiceError, _service_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unexpected_error)
    return app


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": {"code": code, "message": message}}
    )


# Starlette types every handler as (Request, Exception); each narrows its own type.


async def _service_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ServiceError)
    return _error(_HTTP_STATUS.get(type(exc), status.HTTP_400_BAD_REQUEST), exc.code, str(exc))


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
    )
    return _error(status.HTTP_422_UNPROCESSABLE_CONTENT, "validation_error", problems)


async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    _log.error("unhandled_api_error", path=request.url.path, exc_info=exc)
    return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", "internal server error")


app = create_app()
