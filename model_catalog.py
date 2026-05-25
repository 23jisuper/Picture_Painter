"""前端可选模型目录：从 config 读取，缺省时按当前配置生成默认项。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    backend: str
    model: str | None = None


def _parse_catalog(raw_list: Any, fallback: list[ModelOption]) -> list[ModelOption]:
    if not isinstance(raw_list, list) or not raw_list:
        return fallback
    out: list[ModelOption] = []
    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or f"model-{i}").strip()
        label = str(item.get("label") or mid).strip()
        backend = str(item.get("backend") or "").strip()
        if not mid or not backend:
            continue
        model = item.get("model")
        out.append(
            ModelOption(
                id=mid,
                label=label,
                backend=backend,
                model=str(model).strip() if model else None,
            )
        )
    return out or fallback


def build_caption_catalog(raw_vc: dict[str, Any], _default_backend: str, default_oa_model: str) -> list[ModelOption]:
    fallback = [
        ModelOption(id="local_qwen", label="本地 Qwen VL", backend="local"),
        ModelOption(
            id="openai_compatible",
            label=f"OpenAI 兼容 · {default_oa_model}",
            backend="openai_compatible",
            model=default_oa_model,
        ),
    ]
    return _parse_catalog(raw_vc.get("model_catalog"), fallback)


def build_image_catalog(
    raw_ig: dict[str, Any],
    _default_backend: str,
    doubao_model: str,
    minimax_model: str,
) -> list[ModelOption]:
    fallback = [
        ModelOption(
            id="doubao_default",
            label=f"豆包 Seedream · {doubao_model}",
            backend="doubao_ark",
            model=doubao_model,
        ),
        ModelOption(
            id="minimax_default",
            label=f"MiniMax · {minimax_model}",
            backend="minimax_api",
            model=minimax_model,
        ),
    ]
    return _parse_catalog(raw_ig.get("model_catalog"), fallback)


def find_option(options: list[ModelOption], option_id: str | None) -> ModelOption | None:
    if not option_id:
        return options[0] if options else None
    for o in options:
        if o.id == option_id:
            return o
    return options[0] if options else None


def options_payload(options: list[ModelOption]) -> list[dict[str, str | None]]:
    return [{"id": o.id, "label": o.label, "backend": o.backend, "model": o.model} for o in options]
