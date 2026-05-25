"""MiniMax 文生图（含 subject_reference）。"""
from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path

import requests

from settings import AppSettings, MiniMaxSettings

logger = logging.getLogger("picture_painter.minimax")


def _reference_image_value(image_path: Path, mm: MiniMaxSettings) -> str:
    if mm.reference_image_encoding == "url":
        raise RuntimeError(
            "reference_image_encoding 为 url 时，请先将参考图托管为公网 URL，并在前端改为传 reference_image_url（后续可扩展）。"
        )
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/jpeg"
    b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def compose_image_prompt(app: AppSettings, caption: str) -> str:
    t = app.prompt_composition.image_gen_user_template
    if "{caption}" in t:
        return t.replace("{caption}", caption.strip())
    return f"{caption.strip()}\n{t}"


def generate_with_minimax(
    *,
    app: AppSettings,
    caption: str,
    reference_image_path: Path,
    aspect_ratio: str | None = None,
    image_count: int = 1,
    model_override: str | None = None,
) -> list[str]:
    mm = app.image_generation.minimax
    api_key = (os.environ.get(mm.api_key_env) or "").strip() or (mm.api_key or "").strip()
    if not api_key:
        raise RuntimeError(
            "MiniMax 未配置密钥："
            f"请将 image_generation.minimax.api_key_env 设为环境变量名（例如 MINIMAX_API_KEY），"
            f"并在系统中 export 该变量；或在 minimax.api_key 中填写密钥。"
            f"不要把密钥字符串填在 api_key_env 里。"
        )

    prompt = compose_image_prompt(app, caption)
    image_file = _reference_image_value(reference_image_path, mm)
    ar = (aspect_ratio or "").strip() or mm.aspect_ratio
    model = (model_override or "").strip() or mm.model
    n = max(1, min(9, int(image_count)))
    payload = {
        "model": model,
        "prompt": prompt,
        "aspect_ratio": ar,
        "n": n,
        "subject_reference": [{"type": mm.subject_reference_type, "image_file": image_file}],
        "response_format": mm.response_format,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    logger.info(
        "HTTP POST url=%s model=%s aspect_ratio=%s n=%s prompt_len=%s ref_encoding=%s",
        mm.url,
        model,
        ar,
        n,
        len(prompt),
        mm.reference_image_encoding,
    )
    r = requests.post(mm.url, headers=headers, json=payload, timeout=300)
    logger.info("MiniMax 响应 HTTP %s", r.status_code)
    r.raise_for_status()
    body = r.json()
    try:
        images = body["data"]["image_base64"]
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"MiniMax 返回异常: {body}") from e
    if not isinstance(images, list):
        images = [images]
    return [str(x) for x in images]
