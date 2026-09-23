"""Structured errors and HTTP mapping for the OpenAI-compatible API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse


@dataclass
class ServiceError(Exception):
    message: str
    status_code: int = 500
    error_type: str = "server_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ModelError(ServiceError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 404,
        code: str = "model_error",
        param: Optional[str] = "model",
    ) -> None:
        super().__init__(message, status_code, "invalid_request_error", param, code)


class DeviceError(ServiceError):
    def __init__(self, message: str, *, status_code: int = 503, code: str = "device_error") -> None:
        super().__init__(message, status_code, "server_error", None, code)


class ValidationError(ServiceError):
    def __init__(self, message: str, *, param: Optional[str] = None, code: str = "invalid_request") -> None:
        super().__init__(message, 400, "invalid_request_error", param, code)


class QueueFullError(ServiceError):
    def __init__(self, message: str = "request queue is full") -> None:
        super().__init__(message, 429, "rate_limit_error", None, "queue_full")


class ConflictError(ServiceError):
    def __init__(self, message: str, *, code: str = "resource_conflict") -> None:
        super().__init__(message, 409, "invalid_request_error", None, code)


def error_payload(error: ServiceError) -> dict:
    body = {
        "message": error.message,
        "type": error.error_type,
    }
    if error.param is not None:
        body["param"] = error.param
    if error.code is not None:
        body["code"] = error.code
    return {"error": body}


def service_error_response(error: ServiceError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content=error_payload(error))


def install_error_handlers(app) -> None:
    @app.exception_handler(ServiceError)
    async def handle_service_error(_request: Request, exc: ServiceError) -> JSONResponse:
        return service_error_response(exc)
