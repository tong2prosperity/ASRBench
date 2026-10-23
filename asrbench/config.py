"""Local configuration. Secrets are resolved only when a model is executed."""

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

SECRET_KEYS = {
    "api_key",
    "app_id",
    "app_key",
    "access_key",
    "secret_id",
    "secret_key",
    "token",
    "authorization",
}
PROVIDERS = (
    "volcengine",
    "funasr",
    "qwen_asr",
    "paraformer",
    "tencent",
    "tencent_v2",
    "eot_chat",
)
STREAMING = {"tencent", "tencent_v2"}


def redact(value):
    if isinstance(value, dict):
        return {
            k: "<redacted>" if k.lower() in SECRET_KEYS else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def load_config(path="config.yaml"):
    path = Path(path).expanduser().resolve()
    load_dotenv(path.parent / ".env", override=False)
    if not path.exists():
        raise ValueError(f"配置不存在：{path}；请复制 config.yaml.example")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        raise ValueError("YAML 配置格式错误，请检查缩进与字段") from None
    if not isinstance(cfg, dict):
        raise ValueError("配置必须为 YAML 对象")
    profiles = cfg.get("models")
    if profiles is None:
        profiles = [
            dict(profile_id=k, name=k, provider=k, **v)
            for k, v in cfg.get("providers", {}).items()
        ]
    if not isinstance(profiles, list):
        raise ValueError("models 必须为配置列表")
    ids = set()
    for p in profiles:
        if not isinstance(p, dict) or not isinstance(p.get("params", {}), dict):
            raise ValueError("模型配置及 params 必须为对象")
        pid = p.get("profile_id", "")
        if (
            not isinstance(pid, str)
            or not re.fullmatch(r"[a-zA-Z0-9_-]+", pid)
            or pid in ids
        ):
            raise ValueError("profile_id 必须唯一且只包含字母、数字、下划线或连字符")
        ids.add(pid)
        if p.get("provider") not in PROVIDERS:
            raise ValueError(f"不支持的 provider：{p.get('provider')}")
        p.setdefault("name", pid)
        p.setdefault("enabled", True)
    return {**cfg, "models": profiles, "_path": str(path)}


def resolve_profile(profile):
    def resolve(value):
        if isinstance(value, str):

            def sub(match):
                name = match.group(1)
                if not os.environ.get(name):
                    raise ValueError(f"缺少环境变量：{name}")
                return os.environ[name]

            return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", sub, value)
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        return value

    result = resolve(profile.get("params", {}))
    result.update(
        resolve(
            {
                k: v
                for k, v in profile.items()
                if k not in {"params", "profile_id", "name", "provider", "enabled"}
            }
        )
    )
    return result


def public_profiles(cfg):
    return [
        dict(
            redact(p),
            modes=["eot"]
            if p["provider"] == "eot_chat"
            else ["batch", "stream"]
            if p["provider"] in STREAMING
            else ["batch"],
        )
        for p in cfg["models"]
        if p.get("enabled", True)
    ]
