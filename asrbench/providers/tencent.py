"""Original Tencent engine using the shared cancellable socket transport."""

from .tencent_v2 import TencentV2Provider


class TencentProvider(TencentV2Provider):
    name = "tencent"

    def __init__(self, config):
        super().__init__({"engine_model_type": "16k_zh_en", **config})
