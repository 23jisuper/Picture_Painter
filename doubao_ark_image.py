"""火山方舟豆包 Seedream 文生图 / 图生图（OpenAI 兼容客户端）。"""
from __future__ import annotations

import base64
import copy
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

from minimax_image import compose_image_prompt
from settings import AppSettings, DoubaoArkSettings

logger = logging.getLogger("picture_painter.doubao_ark")


def _reference_image_value(image_path: Path, cfg: DoubaoArkSettings) -> str:
    if cfg.reference_image_encoding == "url":
        raise RuntimeError(
            "doubao_ark.reference_image_encoding 为 url 时，请提供可访问的公网图 URL（当前仅支持本地上传转 data_uri）。"
        )
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/jpeg"
    b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _images_to_base64_list(resp, response_format: str) -> list[str]:
    data = getattr(resp, "data", None) or []
    out: list[str] = []
    for item in data:
        if response_format == "b64_json":
            b64 = getattr(item, "b64_json", None)
            if b64:
                out.append(str(b64))
            continue
        url = getattr(item, "url", None)
        if not url:
            continue
        r = requests.get(str(url), timeout=120)
        r.raise_for_status()
        out.append(base64.standard_b64encode(r.content).decode("ascii"))
    return out


def _call_generate(
    *,
    client: OpenAI,
    cfg: DoubaoArkSettings,
    model: str,
    prompt: str,
    extra: dict[str, Any],
    sdk_n: int,
) -> list[str]:
    rf = cfg.response_format
    resp = client.images.generate(
        model=model,
        prompt=prompt,
        size=cfg.size,
        n=sdk_n,
        response_format=rf,
        extra_body=extra,
    )
    return _images_to_base64_list(resp, rf)


def generate_with_doubao_ark(
    *,
    app: AppSettings,
    caption: str,
    reference_image_path: Path,
    image_count: int = 1,
    model_override: str | None = None,
) -> list[str]:
    cfg = app.image_generation.doubao_ark
    api_key = (os.environ.get(cfg.api_key_env) or "").strip() or (cfg.api_key or "").strip()
    if not api_key:
        raise RuntimeError(
            f"豆包 Ark 未配置密钥：请设置环境变量 {cfg.api_key_env}，"
            f"或在 config.yaml 的 image_generation.doubao_ark.api_key 中填写（勿提交仓库）。"
        )

    prompt = compose_image_prompt(app, caption)
    model = (model_override or "").strip() or cfg.model
    target = max(1, min(8, int(image_count)))

    base_extra: dict[str, Any] = dict(cfg.extra_body)
    base_extra["watermark"] = cfg.watermark
    if cfg.send_reference_image:
        base_extra["image"] = _reference_image_value(reference_image_path, cfg)

    client = OpenAI(api_key=api_key, base_url=cfg.base_url)

    def _run_once(*, use_sequential: bool, max_images: int) -> list[str]:
        extra = copy.deepcopy(base_extra)
        if use_sequential and max_images > 1:
            extra["sequential_image_generation"] = "auto"
            extra["sequential_image_generation_options"] = {"max_images": max_images}
            sdk_n = 1
        else:
            extra.pop("sequential_image_generation", None)
            extra.pop("sequential_image_generation_options", None)
            sdk_n = 1 if max_images <= 1 else min(4, max_images)
        logger.info(
            "OpenAI-compatible images.generate model=%s size=%s target=%s sequential=%s sdk_n=%s extra_keys=%s",
            model,
            cfg.size,
            max_images,
            use_sequential and max_images > 1,
            sdk_n,
            sorted(extra.keys()),
        )
        return _call_generate(
            client=client,
            cfg=cfg,
            model=model,
            prompt=prompt,
            extra=extra,
            sdk_n=sdk_n,
        )

    collected: list[str] = []

    if target > 1:
        batch = _run_once(use_sequential=True, max_images=target)
        collected.extend(batch)
        logger.info("豆包 Ark 组图模式首轮返回 %s 张，目标 %s 张", len(batch), target)

    if len(collected) < target:
        need = target - len(collected)
        singles = _run_once(use_sequential=False, max_images=need)
        for img in singles:
            if len(collected) >= target:
                break
            collected.append(img)
        logger.info("豆包 Ark 补量单图模式追加后共 %s 张", len(collected))

    if not collected:
        raise RuntimeError("豆包 Ark 未返回图片数据")
    logger.info("豆包 Ark 完成 张数=%s", len(collected))
    return collected[:target]
