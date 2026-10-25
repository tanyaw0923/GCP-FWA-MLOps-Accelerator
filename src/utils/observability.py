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

    On GCP, google-cloud-logging sends Python log records to
    Cloud Logging. If the package or credentials are unavailable
    (for example in a lightweight local environment), logging
    falls back to stdout without breaking the workload.

    Do not put claim-level payloads, member identifiers, or PHI
    into json_fields.
    """

    logger = logging.getLogger(
        service_name
    )

    logger.setLevel(
        logging.INFO
    )

    if getattr(
        logger,
        "_fwa_observability_configured",
        False,
    ):
        return logger

    try:
        import google.cloud.logging

        cloud_logging_client = (
            google.cloud.logging.Client(
                project=project_id
            )
        )

        cloud_logging_client.setup_logging(
            log_level=logging.INFO
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
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format=(
                    "%(asctime)s "
                    "%(levelname)s "
                    "%(name)s "
                    "%(message)s"
                ),
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

    Cloud Logging recognizes json_fields when using the Google
    Cloud Logging Python handler. Values should be operational
    metadata only, not raw claim/member payloads.
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
