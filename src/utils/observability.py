import json
import logging
import os
from typing import Any, Dict, Optional


def configure_observability(
    project_id: str,
    service_name: str,
) -> logging.Logger:
    """
    Configure operational logging.

    Structured operational events are sent directly to Cloud
    Logging with log_struct(), so they do not clutter stdout and
    do not rely on a background logging thread.

    If Cloud Logging is unavailable, events fall back to a normal
    Python StreamHandler.

    Do not put claim-level payloads, member identifiers, or PHI
    into structured log fields.
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
        import google.cloud.logging

        client = (
            google.cloud.logging.Client(
                project=project_id
            )
        )

        cloud_logger = client.logger(
            service_name
        )

        logger._fwa_cloud_logger = (
            cloud_logger
        )

        logger._fwa_cloud_logging_enabled = (
            True
        )

        cloud_logger.log_struct(
            {
                "event": "cloud_logging_enabled",
                "service": service_name,
                "project_id": project_id,
                "logging_mode": "direct_log_struct",
                "environment": (
                    os.environ.get(
                        "K_SERVICE",
                        "interactive_or_vertex",
                    )
                ),
            },
            severity="INFO",
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

        logger._fwa_cloud_logger = None
        logger._fwa_cloud_logging_enabled = (
            False
        )

        logger.warning(
            "Cloud Logging unavailable; "
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

    When Cloud Logging is available, the event is sent directly
    with log_struct() and is not printed to stdout. Otherwise it
    falls back to standard Python logging.
    """

    payload = {
        "event": event,
    }

    if fields:
        payload.update(
            fields
        )

    cloud_logger = getattr(
        logger,
        "_fwa_cloud_logger",
        None,
    )

    cloud_enabled = getattr(
        logger,
        "_fwa_cloud_logging_enabled",
        False,
    )

    if (
        cloud_enabled
        and cloud_logger is not None
    ):
        try:
            cloud_logger.log_struct(
                payload,
                severity=severity.upper(),
            )

            return

        except Exception as exc:
            logger.warning(
                "Cloud Logging event failed; "
                "falling back to local logging. "
                "Event=%s Reason=%s",
                event,
                exc,
            )

    log_method = getattr(
        logger,
        severity.lower(),
        logger.info,
    )

    log_method(
        "%s %s",
        event,
        json.dumps(
            payload,
            default=str,
            sort_keys=True,
        ),
    )
