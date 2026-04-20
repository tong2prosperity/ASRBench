from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ASRResult:
    text: str
    duration_ms: float
    audio_duration_ms: float = 0.0
    raw_response: dict = field(default_factory=dict)


class ASRProvider(ABC):
    name: str = ""
    supported_formats: frozenset[str] = frozenset({"pcm"})

    @abstractmethod
    async def recognize(self, audio_data: bytes, audio_format: str, sample_rate: int) -> ASRResult:
        ...

    async def close(self):
        pass
