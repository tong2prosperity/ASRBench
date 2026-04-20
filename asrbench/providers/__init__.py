from .base import ASRProvider, ASRResult
from .volcengine import VolcEngineProvider
from .funasr import FunASRProvider
from .qwen_asr import QwenASRProvider
from .paraformer import ParaformerProvider
from .tencent import TencentProvider

PROVIDER_REGISTRY: dict[str, type[ASRProvider]] = {
    "volcengine": VolcEngineProvider,
    "funasr": FunASRProvider,
    "qwen_asr": QwenASRProvider,
    "paraformer": ParaformerProvider,
    "tencent": TencentProvider,
}
