"""从 YAML 加载应用配置；缺省键使用内置默认值。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml


@dataclass
class ServerSettings:
    host: str
    port: int
    upload_dir: str


@dataclass
class PathsSettings:
    checkpoint_relative: str


@dataclass
class VisionLocalSettings:
    min_pixels: int
    max_pixels: int
    max_new_tokens: int


@dataclass
class VisionOpenAISettings:
    base_url: str
    model: str
    api_key_env: str
    max_tokens: int


@dataclass
class VisionCaptionSettings:
    backend: Literal["local", "openai_compatible"]
    prompt_template: str
    local: VisionLocalSettings
    openai_compatible: VisionOpenAISettings


@dataclass
class MiniMaxSettings:
    url: str
    model: str
    aspect_ratio: str
    response_format: str
    # 读取 os.environ[api_key_env]；api_key 为可选直配（勿提交仓库）
    api_key_env: str
    api_key: str
    subject_reference_type: str
    reference_image_encoding: Literal["data_uri", "url"]


@dataclass
class DoubaoArkSettings:
    """火山方舟豆包 Seedream 等文生图 / 图生图（OpenAI 兼容 SDK）。"""
    base_url: str
    api_key_env: str
    api_key: str
    model: str
    size: str
    response_format: Literal["url", "b64_json"]
    watermark: bool
    reference_image_encoding: Literal["data_uri", "url"]
    send_reference_image: bool
    extra_body: dict[str, Any]


@dataclass
class ImageGenerationSettings:
    backend: Literal["minimax_api", "doubao_ark", "none"]
    minimax: MiniMaxSettings
    doubao_ark: DoubaoArkSettings


@dataclass
class PromptCompositionSettings:
    image_gen_user_template: str


@dataclass
class AppSettings:
    server: ServerSettings
    paths: PathsSettings
    vision_caption: VisionCaptionSettings
    image_generation: ImageGenerationSettings
    prompt_composition: PromptCompositionSettings


def _deep_get(d: dict[str, Any], *keys: str, default: Any) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_app_settings(config_path: Path) -> AppSettings:
    raw: dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    server = ServerSettings(
        host=str(_deep_get(raw, "server", "host", default="127.0.0.1")),
        port=int(_deep_get(raw, "server", "port", default=8000)),
        upload_dir=str(_deep_get(raw, "server", "upload_dir", default="temp_uploads")),
    )
    paths = PathsSettings(
        checkpoint_relative=str(
            _deep_get(raw, "paths", "checkpoint_relative", default="checkpoint/xy_model")
        ),
    )
    vl = raw.get("vision_caption") or {}
    backend = str(vl.get("backend", "local"))
    if backend not in ("local", "openai_compatible"):
        backend = "local"
    loc = vl.get("local") or {}
    oa = vl.get("openai_compatible") or {}
    prompt_template = str(
        vl.get("prompt_template")
        or loc.get(
            "prompt_template",
            "图中人物性别或身份为：{gender}。请以专业修图师角度描述画面并给出可直接使用的修图提示词。",
        )
    )
    vision_caption = VisionCaptionSettings(
        backend=backend,  # type: ignore[arg-type]
        prompt_template=prompt_template,
        local=VisionLocalSettings(
            min_pixels=int(loc.get("min_pixels", 256 * 28 * 28)),
            max_pixels=int(loc.get("max_pixels", 512 * 28 * 28)),
            max_new_tokens=int(loc.get("max_new_tokens", 512)),
        ),
        openai_compatible=VisionOpenAISettings(
            base_url=str(oa.get("base_url", "https://api.openai.com/v1")).rstrip("/"),
            model=str(oa.get("model", "gpt-4o-mini")),
            api_key_env=str(oa.get("api_key_env", "OPENAI_API_KEY")),
            max_tokens=int(oa.get("max_tokens", 1024)),
        ),
    )
    ig = raw.get("image_generation") or {}
    ig_backend = str(ig.get("backend", "minimax_api"))
    if ig_backend not in ("minimax_api", "doubao_ark", "none"):
        ig_backend = "minimax_api"
    mm = ig.get("minimax") or {}
    ref_enc = str(mm.get("reference_image_encoding", "data_uri"))
    if ref_enc not in ("data_uri", "url"):
        ref_enc = "data_uri"
    api_key_env = str(mm.get("api_key_env", "MINIMAX_API_KEY"))
    api_key_inline = str(mm.get("api_key", "")).strip()
    # 误把密钥写在 api_key_env：该字段只能是环境变量名（如 MINIMAX_API_KEY）
    if not api_key_inline and api_key_env.startswith("sk-"):
        api_key_inline = api_key_env.strip()
        api_key_env = "MINIMAX_API_KEY"
    db = ig.get("doubao_ark") or {}
    db_api_key_env = str(db.get("api_key_env", "ARK_API_KEY"))
    db_api_key_inline = str(db.get("api_key", "")).strip()
    if not db_api_key_inline and db_api_key_env.startswith("sk-"):
        db_api_key_inline = db_api_key_env.strip()
        db_api_key_env = "ARK_API_KEY"
    db_rf = str(db.get("response_format", "url"))
    if db_rf not in ("url", "b64_json"):
        db_rf = "url"
    db_ref = str(db.get("reference_image_encoding", "data_uri"))
    if db_ref not in ("data_uri", "url"):
        db_ref = "data_uri"
    db_extra = db.get("extra_body")
    if not isinstance(db_extra, Mapping):
        db_extra = {}
    doubao_ark = DoubaoArkSettings(
        base_url=str(db.get("base_url", "https://ark.cn-beijing.volces.com/api/v3")).rstrip("/"),
        api_key_env=db_api_key_env,
        api_key=db_api_key_inline,
        model=str(db.get("model", "doubao-seedream-4-0-250828")),
        size=str(db.get("size", "2K")),
        response_format=db_rf,  # type: ignore[arg-type]
        watermark=bool(db.get("watermark", True)),
        reference_image_encoding=db_ref,  # type: ignore[arg-type]
        send_reference_image=bool(db.get("send_reference_image", True)),
        extra_body=dict(db_extra),
    )
    image_generation = ImageGenerationSettings(
        backend=ig_backend,  # type: ignore[arg-type]
        minimax=MiniMaxSettings(
            url=str(mm.get("url", "https://api.minimaxi.com/v1/image_generation")),
            model=str(mm.get("model", "image-01")),
            aspect_ratio=str(mm.get("aspect_ratio", "16:9")),
            response_format=str(mm.get("response_format", "base64")),
            api_key_env=api_key_env,
            api_key=api_key_inline,
            subject_reference_type=str(mm.get("subject_reference_type", "character")),
            reference_image_encoding=ref_enc,  # type: ignore[arg-type]
        ),
        doubao_ark=doubao_ark,
    )
    pc = raw.get("prompt_composition") or {}
    prompt_composition = PromptCompositionSettings(
        image_gen_user_template=str(
            pc.get(
                "image_gen_user_template",
                "{caption}\n"
                "第二幅参考图为人物主体：须严格保留面部容貌与五官特征，禁止换脸；"
                "仅调整服饰、背景与光影氛围以呼应上文描述。",
            )
        ),
    )
    return AppSettings(
        server=server,
        paths=paths,
        vision_caption=vision_caption,
        image_generation=image_generation,
        prompt_composition=prompt_composition,
    )


def resolve_config_path(base_dir: Path, exe_dir: Path | None, frozen: bool) -> Path:
    env = os.environ.get("PICTURE_PAINTER_CONFIG")
    if env:
        return Path(env)
    if frozen and exe_dir is not None:
        p = exe_dir / "config.yaml"
        if p.is_file():
            return p
    return base_dir / "config.yaml"
