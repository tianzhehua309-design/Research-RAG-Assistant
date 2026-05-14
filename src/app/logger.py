import json
import logging
from datetime import datetime,timezone
from encodings import palmos

from pip._internal.utils import retry


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp" : datetime.now(timezone.utc).isoformat(),
            "level" : record.levelname,
            "logger" : record.name,
            "message" : record.getMessage(),
        }

        for field in (
            "event",
            "request_id",
            "error_code",
            "retryable",
            "method",
            "path",
            "status_code",
            "duration_ms",
        ):
            if hasattr(record, field):
                payload[field] = getattr(record, field)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # 如果不写 ensure_ascii=False，中文会变成 Unicode 编码：
        return json.dumps(payload,ensure_ascii=False)

def get_logger(name: str = "app") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    logger.addHandler(handler)
    logger.propagate = False

    return logger