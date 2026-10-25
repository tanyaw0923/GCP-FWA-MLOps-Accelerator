import json
import logging
import os
from typing import Any, Dict, Optional


def configure_observability(
    project_id: str,
    service_name: str,
) -> logging.Logger:
    """
    Configure structured operational logging.

    On GCP, StructuredLogHandler writes structured JSON to stdout
    so managed runtimes can collect it without a background log
    transport thread. This avoids shutdown-time log loss from the
    asynchronous CloudLoggingHandler.

    If google-cloud-logging is unavailable, logging falls back to
    the standard Python StreamHandler.

    Do not put claim-level payloads, member identifiers, or PHI
    into json_fields.
    """

    logger = logging.getLogger(
        service_name
    )

    logger.setLevel(
        logging.INFO
    )

    logger.propagate = False

    if getattr(
        logger,
        "_fwa_observability_configured",
        False,
    ):
        return logger

    try:
        from google.cloud.logging_v2.handlers import (
            StructuredLogHandler,
        )

        handler = StructuredLogHandler()

        handler.setLevel(
            logging.INFO
        )

        logger.addHandler(
            handler
        )

        logger.info(
            "cloud_logging_enabled",
            extra={
                "json_fields": {
                    "event": (
                        "cloud_logging_enabled"
                    ),
                    "service": service_name,
                    "project_id": project_id,
                    "logging_handler": (
                        "StructuredLogHandler"
                    ),
                    "environment": (
                        os.environ.get(
                            "K_SERVICE",
                            "interactive_or_vertex",
                        )
                    ),
                }
            },
        )

    except Exception as exc:
        handler = logging.StreamHandler()

        handler.setLevel(
            logging.INFO
        )

        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s "
                "%(levelname)s "
                "%(name)s "
                "%(message)s"
            )
        )

        logger.addHandler(
            handler
        )

        logger.warning(
            "Structured Cloud Logging unavailable; "
            "using standard Python logging. "
            "Reason=%s",
            exc,
        )

    logger._fwa_observability_configured = True

    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    severity: str = "INFO",
    fields: Optional[
        Dict[str, Any]
    ] = None,
) -> None:
    """
    Write one structured operational event.

    StructuredLogHandler emits json_fields as structured JSON to
    stdout, which managed GCP runtimes can ingest into Cloud
    Logging. Values should be operational metadata only, not raw
    claim/member payloads.
    """

    payload = {
        "event": event,
    }

    if fields:
        payload.update(
            fields
        )

    log_method = getattr(
        logger,
        severity.lower(),
        logger.info,
    )

    try:
        log_method(
            event,
            extra={
                "json_fields": payload
            },
        )

    except Exception:
        log_method(
            "%s %s",
            event,
            json.dumps(
                payload,
                default=str,
                sort_keys=True,
            ),
        )
