from dotenv import load_dotenv
import os
import sys
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import cv2
import asyncio
# from paddleocr import PaddleOCR
import torch
from PIL import Image, ImageFile
from io import BytesIO
from pydantic import BaseModel
from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR # Paddle的cuda镜像太大，改用torch，RapidOCR支持torch
import cn_clip.clip as clip
ImageFile.LOAD_TRUNCATED_IMAGES = True

on_linux = sys.platform.startswith('linux')

load_dotenv()
app = FastAPI()

api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
http_port = int(os.getenv("HTTP_PORT", "8060"))
server_restart_time = int(os.getenv("SERVER_RESTART_TIME", "300"))
env_auto_load_txt_modal = os.getenv("AUTO_LOAD_TXT_MODAL", "off") == "on" # 是否自动加载CLIP文本模型，开启可以优化第一次搜索时的响应速度,文本模型占用700多m内存

# RapidOCR 详细配置环境变量
# Det 配置
ocr_det_lang_type = os.getenv("OCR_DET_LANG_TYPE", "CH")  # 检测模型语言: CH, EN, MULTI
ocr_det_model_type = os.getenv("OCR_DET_MODEL_TYPE", "mobile")  # 检测模型类型: mobile, server
ocr_det_ocr_version = os.getenv("OCR_DET_OCR_VERSION", "PPOCRV4")  # 检测模型版本: PPOCRV4, PPOCRV5

# Rec 配置
ocr_rec_lang_type = os.getenv("OCR_REC_LANG_TYPE", "CH")  # 识别模型语言: CH, EN, JAPAN, KOREAN 等
ocr_rec_model_type = os.getenv("OCR_REC_MODEL_TYPE", "mobile")  # 识别模型类型: mobile, server
ocr_rec_ocr_version = os.getenv("OCR_REC_OCR_VERSION", "PPOCRV4")  # 识别模型版本: PPOCRV4, PPOCRV5

# Cls 配置
ocr_cls_lang_type = os.getenv("OCR_CLS_LANG_TYPE", "CH")  # 分类模型语言: CH, EN, MULTI
ocr_cls_model_type = os.getenv("OCR_CLS_MODEL_TYPE", "mobile")  # 分类模型类型: mobile, server
ocr_cls_ocr_version = os.getenv("OCR_CLS_OCR_VERSION", "PPOCRV4")  # 分类模型版本: PPOCRV4, PPOCRV5
ocr_CUDA_device_id = int(os.getenv("OCR_CUDA_DEVICE_ID", "0"))  # 指定GPU id，强制转为数字

# 辅助函数：字符串转枚举
def get_lang_det(value: str) -> LangDet:
    """将字符串转换为 LangDet 枚举"""
    mapping = {e.name: e for e in LangDet}
    return mapping.get(value.upper(), LangDet.CH)

def get_lang_rec(value: str) -> LangRec:
    """将字符串转换为 LangRec 枚举"""
    mapping = {e.name: e for e in LangRec}
    return mapping.get(value.upper(), LangRec.CH)

def get_model_type(value: str) -> ModelType:
    """将字符串转换为 ModelType 枚举"""
    mapping = {e.name: e for e in ModelType}
    return mapping.get(value.upper(), ModelType.MOBILE)

def get_ocr_version(value: str) -> OCRVersion:
    """将字符串转换为 OCRVersion 枚举"""
    mapping = {e.name: e for e in OCRVersion}
    return mapping.get(value.upper(), OCRVersion.PPOCRV4)


clip_model_name = os.getenv("CLIP_MODEL")


ocr_model = None
clip_processor = None
clip_model = None

restart_task = None
restart_lock = asyncio.Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"

class ClipTxtRequest(BaseModel):
    text: str

def load_ocr_model():
    global ocr_model
    if ocr_model is None:
        ocr_model = RapidOCR(
            params={
                "Det.engine_type": EngineType.TORCH,
                "Cls.engine_type": EngineType.TORCH,
                "Rec.engine_type": EngineType.TORCH,
                "EngineConfig.torch.use_cuda": True,  # 使用 torch GPU 版推理
                "EngineConfig.torch.cuda_ep_cfg.device_id": ocr_CUDA_device_id,  # 指定GPU id
                "Det.lang_type": get_lang_det(ocr_det_lang_type),
                "Det.model_type": get_model_type(ocr_det_model_type),
                "Det.ocr_version": get_ocr_version(ocr_det_ocr_version),
                "Rec.lang_type": get_lang_rec(ocr_rec_lang_type),
                "Rec.model_type": get_model_type(ocr_rec_model_type),
                "Rec.ocr_version": get_ocr_version(ocr_rec_ocr_version),
                "Cls.lang_type": get_lang_det(ocr_cls_lang_type),
                "Cls.model_type": get_model_type(ocr_cls_model_type),
                "Cls.ocr_version": get_ocr_version(ocr_cls_ocr_version),
            }
        )
        # https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/

def load_clip_model():
    global clip_processor
    global clip_model
    if clip_processor is None:
        model, preprocess = clip.load_from_name(clip_model_name, device=device)
        model.eval()
        clip_model = model
        clip_processor = preprocess

@app.on_event("startup")
async def startup_event():
    if env_auto_load_txt_modal:
        load_clip_model()

@app.on_event("shutdown")
async def on_shutdown():
    if restart_task and not restart_task.done():
        restart_task.cancel()
        try:
            await restart_task
        except asyncio.CancelledError:
            pass

async def restart_timer():
    await asyncio.sleep(server_restart_time)
    restart_program()

@app.middleware("http")
async def activity_monitor(request, call_next):
    global restart_task

    async with restart_lock:
        if restart_task and not restart_task.done():
            restart_task.cancel()

        restart_task = asyncio.create_task(restart_timer())

    response = await call_next(request)
    return response


async def verify_header(api_key: str = Header(...)):
    # 在这里编写验证逻辑，例如检查 api_key 是否有效
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


def to_fixed(num):
    return str(round(num, 2))

def convert_rapidocr_to_json(rapidocr_output):

    if rapidocr_output.txts is None:
        return {'texts': [], 'scores': [], 'boxes': []}

    texts = list(rapidocr_output.txts)
    scores = [f"{score:.2f}" for score in rapidocr_output.scores]
    boxes_coords = rapidocr_output.boxes

    boxes = []
    for box in boxes_coords:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        width = x_max - x_min
        height = y_max - y_min

        boxes.append({
            'x': to_fixed(x_min),
            'y': to_fixed(y_min),
            'width': to_fixed(width),
            'height': to_fixed(height)
        })

    output = {
        'texts': texts,
        'scores': scores,
        'boxes': boxes
    }

    return output

@app.get("/", response_class=HTMLResponse)
async def top_info():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MT Photos AI Server</title>
    <style>p{text-align: center;}</style>
</head>
<body>
<p style="font-weight: 600;">MT Photos智能识别服务</p>
<p>服务状态： 运行中</p>
<p>使用方法： <a href="https://mtmt.tech/docs/advanced/ocr_api">https://mtmt.tech/docs/advanced/ocr_api</a></p>
</body>
</html>"""
    return html_content


@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {
        'result': 'pass',
        "title": "mt-photos-ai服务",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
        "device": device
    }


@app.post("/restart")
async def check_req(api_key: str = Depends(verify_header)):
    # cuda版本 OCR没有显存未释放问题，这边可以关闭重启
    return {'result': 'unsupported'}
    # restart_program()

@app.post("/restart_v2")
async def check_req(api_key: str = Depends(verify_header)):
    # 预留触发服务重启接口-自动释放内存
    restart_program()
    return {'result': 'pass'}

@app.post("/ocr")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    load_ocr_model()
    image_bytes = await file.read()
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}

        _result = await predict(ocr_model, img)
        result = convert_rapidocr_to_json(_result)
        del img
        del _result
        return {'result': result}
    except Exception as e:
        print(e)
        return {'result': [], 'msg': str(e)}

@app.post("/clip/img")
async def clip_process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    load_clip_model()
    image_bytes = await file.read()
    try:
        image = clip_processor(Image.open(BytesIO(image_bytes))).unsqueeze(0).to(device)
        image_features = clip_model.encode_image(image)
        return {'result': ["{:.16f}".format(vec) for vec in image_features[0]]}
    except Exception as e:
        print(e)
        return {'result': [], 'msg': str(e)}

@app.post("/clip/txt")
async def clip_process_txt(request:ClipTxtRequest, api_key: str = Depends(verify_header)):
    load_clip_model()
    text = clip.tokenize([request.text]).to(device)
    text_features = clip_model.encode_text(text)
    return {'result': ["{:.16f}".format(vec) for vec in text_features[0]]}

async def predict(predict_func, inputs):
    return await asyncio.get_running_loop().run_in_executor(None, predict_func, inputs)


def restart_program():
    print("restart_program")
    python = sys.executable
    os.execl(python, python, *sys.argv)


if __name__ == "__main__":
    uvicorn.run("server:app", host=None, port=http_port)
