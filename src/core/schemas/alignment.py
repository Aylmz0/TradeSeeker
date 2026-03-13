from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AlignmentError(Enum):
    NONE = "NONE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    API_FAILURE = "API_FAILURE"
    CORRUPT_TIMESTAMPS = "CORRUPT_TIMESTAMPS"
    EXCESSIVE_MISMATCH = "EXCESSIVE_MISMATCH"


@dataclass
class AlignmentResult:
    aligned: bool
    max_delta_seconds: float = 0.0
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    error_type: AlignmentError = AlignmentError.NONE
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "aligned": self.aligned,
            "max_delta_seconds": self.max_delta_seconds,
            "mismatches": self.mismatches,
            "error_type": self.error_type.value,
            "error_message": self.error_message,
        }
