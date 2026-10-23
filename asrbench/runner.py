"""Compatibility entry point. Evaluation is implemented in service.py."""

from .config import load_config as load_config
from .cli import main as cli

if __name__ == "__main__":
    cli()
