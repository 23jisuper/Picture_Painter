"""图生文：本地 Qwen VL 或 OpenAI 兼容多模态 API。"""
from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

import requests
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from qwen_vl_utils import process_vision_info

from settings import AppSettings, VisionCaptionSettings


def build_caption_prompt(settings: VisionCaptionSettings, gender: str) -> str:
    return settings.prompt_template.format(gender=gender)


def caption_local(
    *,
    model,
    processor,
    image_path: Path,
    gender: str,
    cfg: VisionCaptionSettings,
) -> str:
    prompt = build_caption_prompt(cfg, gender)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.local.max_new_tokens,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def caption_openai_compatible(
    *,
    image_path: Path,
    gender: str,
    cfg: VisionCaptionSettings,
    model_override: str | None = None,
) -> str:
    oa = cfg.openai_compatible
    model = (model_override or "").strip() or oa.model
    key = os.environ.get(oa.api_key_env)
    if not key:
        raise RuntimeError(f"未设置环境变量 {oa.api_key_env}，无法调用图生文 API")

    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/jpeg"
    b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    user_prompt = build_caption_prompt(cfg, gender)
    url = f"{oa.base_url}/chat/completions"
    payload = {
        "model": model,
        "max_tokens": oa.max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"图生文 API 返回结构异常: {data}") from e


def run_caption(
    *,
    app: AppSettings,
    model,
    processor,
    image_path: Path,
    gender: str,
    backend_override: str | None = None,
    model_override: str | None = None,
) -> str:
    vc = app.vision_caption
    backend = (backend_override or vc.backend).strip()
    if backend == "local":
        if model is None or processor is None:
            raise RuntimeError("本地图生文模型未加载，请检查 checkpoint 路径或将 vision_caption.backend 改为 openai_compatible")
        return caption_local(model=model, processor=processor, image_path=image_path, gender=gender, cfg=vc)
    if backend == "openai_compatible":
        return caption_openai_compatible(
            image_path=image_path,
            gender=gender,
            cfg=vc,
            model_override=model_override,
        )
    raise RuntimeError(f"不支持的图生文后端: {backend}")
