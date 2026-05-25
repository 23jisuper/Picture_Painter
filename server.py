import sys
import os

# --- 【必加】解决 --noconsole 导致的 NoneType 报错 ---
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import logging
import shutil
import threading
import time
import webbrowser
from pathlib import Path

import yaml
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from caption_backend import run_caption
from doubao_ark_image import generate_with_doubao_ark
from minimax_image import generate_with_minimax
from model_catalog import (
    ModelOption,
    build_caption_catalog,
    build_image_catalog,
    find_option,
    options_payload,
)
from settings import AppSettings, load_app_settings, resolve_config_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("picture_painter")

app = FastAPI()

if getattr(sys, "frozen", False):
    EXE_DIR = Path(sys.executable).parent
    HTML_DIR = Path(sys._MEIPASS)
    sys.stderr = open(EXE_DIR / "error_log.txt", "a")
    sys.stdout = open(EXE_DIR / "output_log.txt", "a")
    BASE_DIR = Path(sys.executable).parent
else:
    EXE_DIR = None
    BASE_DIR = Path(__file__).parent
    HTML_DIR = BASE_DIR

CONFIG_PATH = resolve_config_path(BASE_DIR, EXE_DIR, getattr(sys, "frozen", False))
SETTINGS: AppSettings = load_app_settings(CONFIG_PATH)
logger.info("配置已加载: %s", CONFIG_PATH.resolve())

_RAW_CONFIG: dict = {}
if CONFIG_PATH.is_file():
    with CONFIG_PATH.open("r", encoding="utf-8") as _f:
        _RAW_CONFIG = yaml.safe_load(_f) or {}

_vc_raw = _RAW_CONFIG.get("vision_caption") or {}
_ig_raw = _RAW_CONFIG.get("image_generation") or {}
CAPTION_MODELS: list[ModelOption] = build_caption_catalog(
    _vc_raw,
    SETTINGS.vision_caption.backend,
    SETTINGS.vision_caption.openai_compatible.model,
)
IMAGE_MODELS: list[ModelOption] = build_image_catalog(
    _ig_raw,
    SETTINGS.image_generation.backend,
    SETTINGS.image_generation.doubao_ark.model,
    SETTINGS.image_generation.minimax.model,
)
DEFAULT_CAPTION_MODEL_ID = CAPTION_MODELS[0].id if CAPTION_MODELS else ""
DEFAULT_IMAGE_MODEL_ID = IMAGE_MODELS[0].id if IMAGE_MODELS else ""

# MiniMax 常见比例；前端传入不在此集合时回退为配置默认值
ALLOWED_ASPECT_RATIOS = frozenset(
    {"1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"}
)
ALLOWED_IMAGE_COUNTS = frozenset({1, 2, 4, 8})


def _resolve_image_count(value: str | int | None) -> int:
    default = 1
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    try:
        n = int(value)
    except (TypeError, ValueError):
        logger.warning("【步骤2/文生图】image_count 无效，已回退默认: 收到=%r 使用=%s", value, default)
        return default
    if n not in ALLOWED_IMAGE_COUNTS:
        logger.warning(
            "【步骤2/文生图】image_count 不在允许范围 %s，已回退默认: 收到=%s 使用=%s",
            sorted(ALLOWED_IMAGE_COUNTS),
            n,
            default,
        )
        return default
    return n


def _resolve_aspect_ratio(value: str | None) -> str:
    default = SETTINGS.image_generation.minimax.aspect_ratio
    v = (value or "").strip()
    if v in ALLOWED_ASPECT_RATIOS:
        return v
    if v:
        logger.warning("【步骤2/文生图】aspect_ratio 未在白名单内，已回退默认: 收到=%r 使用=%r", v, default)
    return default


def _workspace_root() -> Path:
    return EXE_DIR if getattr(sys, "frozen", False) and EXE_DIR is not None else BASE_DIR


def _model_root() -> Path:
    if getattr(sys, "frozen", False) and EXE_DIR is not None:
        return EXE_DIR.parent.parent
    return BASE_DIR


def _resolve_path_setting(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (_workspace_root() / path)


UPLOAD_DIR = _resolve_path_setting(SETTINGS.server.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = _model_root() / Path(SETTINGS.paths.checkpoint_relative)

model = None
processor = None


@app.on_event("startup")
def load_model():
    global model, processor
    if SETTINGS.vision_caption.backend != "local":
        logger.info("【启动】图生文后端=%s，跳过本地权重加载", SETTINGS.vision_caption.backend)
        return
    logger.info("【启动】正在加载本地模型: %s", MODEL_ID.resolve())
    if not MODEL_ID.exists():
        logger.error("【启动】checkpoint 不存在: %s", MODEL_ID)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    vl = SETTINGS.vision_caption.local
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=vl.min_pixels,
        max_pixels=vl.max_pixels,
    )
    logger.info("【启动】本地模型就绪 device=%s dtype=%s", device, dtype)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open(HTML_DIR / "index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/options")
async def api_options():
    return JSONResponse(
        {
            "caption_models": options_payload(CAPTION_MODELS),
            "image_models": options_payload(IMAGE_MODELS),
            "default_caption_model_id": DEFAULT_CAPTION_MODEL_ID,
            "default_image_model_id": DEFAULT_IMAGE_MODEL_ID,
            "image_backend": SETTINGS.image_generation.backend,
        }
    )


async def _describe_impl(image: UploadFile, gender: str, caption_model_id: str | None = None):
    fname = image.filename or "upload.bin"
    gpreview = gender[:48] + ("…" if len(gender) > 48 else "")
    logger.info("【步骤1/图生文】收到请求 filename=%s 身份提示(len=%s)=%r", fname, len(gender), gpreview)
    file_path = UPLOAD_DIR / fname
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    logger.info("【步骤1/图生文】已保存临时文件 bytes=%s path=%s", file_path.stat().st_size, file_path.name)
    try:
        cap_opt = find_option(CAPTION_MODELS, caption_model_id)
        cap_backend = cap_opt.backend if cap_opt else SETTINGS.vision_caption.backend
        cap_model = cap_opt.model if cap_opt else None
        logger.info(
            "【步骤1/图生文】开始推理 backend=%s model_id=%s model=%s",
            cap_backend,
            cap_opt.id if cap_opt else None,
            cap_model,
        )
        text = run_caption(
            app=SETTINGS,
            model=model,
            processor=processor,
            image_path=file_path,
            gender=gender,
            backend_override=cap_backend,
            model_override=cap_model,
        )
        preview = (text[:120] + "…") if len(text) > 120 else text
        logger.info("【步骤1/图生文】完成 描述长度=%s 预览=%r", len(text), preview)
        return {"status": "success", "result": text}
    except Exception as e:
        logger.exception("【步骤1/图生文】失败: %s", e)
        return {"status": "error", "message": str(e)}
    finally:
        if file_path.exists():
            os.remove(file_path)
            logger.info("【步骤1/图生文】已删除临时文件")


@app.post("/generate_prompt")
@app.post("/describe")
async def describe_image(
    image: UploadFile = File(...),
    gender: str = Form(...),
    caption_model_id: str | None = Form(None),
):
    return await _describe_impl(image, gender, caption_model_id)


@app.post("/generate_image")
async def generate_image(
    reference_image: UploadFile = File(...),
    caption: str = Form(...),
    aspect_ratio: str | None = Form(None),
    image_count: str | None = Form(None),
    image_model_id: str | None = Form(None),
):
    if not (caption or "").strip():
        logger.warning("【步骤2/文生图】拒绝: caption 为空")
        return {"status": "error", "message": "caption 不能为空"}
    cap = caption.strip()
    ar = _resolve_aspect_ratio(aspect_ratio)
    count = _resolve_image_count(image_count)
    fname = reference_image.filename or "ref.bin"
    logger.info(
        "【步骤2/文生图】收到请求 ref_filename=%s caption_len=%s aspect_ratio=%s image_count=%s backend=%s",
        fname,
        len(cap),
        ar,
        count,
        SETTINGS.image_generation.backend,
    )
    file_path = UPLOAD_DIR / fname
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(reference_image.file, buffer)
    logger.info("【步骤2/文生图】参考图已保存 bytes=%s", file_path.stat().st_size)
    try:
        img_opt = find_option(IMAGE_MODELS, image_model_id)
        if not img_opt:
            return {"status": "error", "message": "未找到所选文生图模型"}
        backend = img_opt.backend
        model_name = img_opt.model
        logger.info(
            "【步骤2/文生图】调用 backend=%s model_id=%s model=%s",
            backend,
            img_opt.id,
            model_name,
        )
        if backend == "minimax_api":
            b64_list = generate_with_minimax(
                app=SETTINGS,
                caption=cap,
                reference_image_path=file_path,
                aspect_ratio=ar,
                image_count=count,
                model_override=model_name,
            )
        elif backend == "doubao_ark":
            b64_list = generate_with_doubao_ark(
                app=SETTINGS,
                caption=cap,
                reference_image_path=file_path,
                image_count=count,
                model_override=model_name,
            )
        else:
            return {"status": "error", "message": f"不支持的文生图后端: {backend}"}
        logger.info("【步骤2/文生图】完成 返回图片张数=%s", len(b64_list))
        return {"status": "success", "images_base64": b64_list}
    except Exception as e:
        logger.exception("【步骤2/文生图】失败: %s", e)
        return {"status": "error", "message": str(e)}
    finally:
        if file_path.exists():
            os.remove(file_path)
            logger.info("【步骤2/文生图】已删除临时参考图")


@app.post("/shutdown")
async def shutdown():
    logger.info("【系统】收到关闭指令，准备退出")

    def kill_process():
        time.sleep(1)
        os._exit(0)

    threading.Thread(target=kill_process).start()
    return {"status": "success", "message": "Server shutting down..."}


def open_browser():
    time.sleep(5)
    webbrowser.open(f"http://{SETTINGS.server.host}:{SETTINGS.server.port}")


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(
        app,
        host=SETTINGS.server.host,
        port=SETTINGS.server.port,
        log_config=None,
    )
