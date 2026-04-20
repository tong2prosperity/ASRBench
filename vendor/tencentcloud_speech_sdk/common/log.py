"""Minimal logger for the vendored SDK.

PATCHED from upstream: the original version calls
`logging.basicConfig(level=logging.DEBUG)` at import time (which hijacks
the root logger for the whole host process) and creates a rotating
`tencent_speech.log` file in cwd. Neither is acceptable for a library.

We replace it with a plain NullHandler logger; the host app can attach
its own handlers if it wants SDK events.
"""
import logging

logger = logging.getLogger("tencent_speech.log")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)
