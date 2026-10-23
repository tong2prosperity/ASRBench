"""Compatibility name for Tencent ASR 2.0 using the shared socket transport."""

from .tencent_v2 import TencentV2Provider


class TencentProvider(TencentV2Provider):
    name = "tencent"

    def __init__(self, config):
        super().__init__({"engine_model_type": "16k_zh_en_2.0", **config})
