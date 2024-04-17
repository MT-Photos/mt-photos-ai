from dotenv import load_dotenv
import os
import sys
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
import uvicorn
import numpy as np
import cv2
import asyncio
from paddleocr import PaddleOCR
import torch
from PIL import Image, ImageFile
from io import BytesIO
from pydantic import BaseModel
import cn_clip.clip as clip
ImageFile.LOAD_TRUNCATED_IMAGES = True

on_linux = sys.platform.startswith('linux')

load_dotenv()
app = FastAPI()
api_auth_key = os.getenv("API_AUTH_KEY")
clip_model_name = os.getenv("CLIP_MODEL")


inactive_task = None
ocr_model = None
clip_processor = None
clip_model = None

device = "cuda" if torch.cuda.is_available() else "cpu"

class ClipTxtRequest(BaseModel):
    text: str

def load_ocr_model():
    global ocr_model
    if ocr_model is None:
        ocr_model = PaddleOCR(
            show_log=False,
            use_angle_cls=True,
            lang="ch",
            ocr_version="PP-OCRv4",
            use_gpu=True
        )
        # https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/whl.md

def load_clip_model():
    global clip_processor
    global clip_model
    if clip_processor is None:
        model, preprocess = clip.load_from_name(clip_model_name, device=device)
        model.eval()
        clip_model = model
        clip_processor = preprocess

async def check_inactive():
    await asyncio.sleep(300)
    restart_program()

@app.middleware("http")
async def check_activity(request, call_next):
    global inactive_task
    if inactive_task:
        inactive_task.cancel()

    inactive_task = asyncio.create_task(check_inactive())
    response = await call_next(request)
    return response


async def verify_header(api_key: str = Header(...)):
    # 在这里编写验证逻辑，例如检查 api_key 是否有效
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


def to_fixed(num):
    return str(round(num, 2))


def trans_result(result):
    texts = []
    scores = []
    boxes = []
    if result is None:
        return {'texts': texts, 'scores': scores, 'boxes': boxes}
    for res_i in result:
        dt_box = res_i[0]
        box = {
            'x': to_fixed(dt_box[0][0]),
            'y': to_fixed(dt_box[0][1]),
            'width': to_fixed(dt_box[1][0] - dt_box[0][0]),
            'height': to_fixed(dt_box[2][1] - dt_box[0][1])
        }
        boxes.append(box)
        scores.append(to_fixed(res_i[1][1]))
        texts.append(res_i[1][0])
    return {'texts': texts, 'scores': scores, 'boxes': boxes}


@app.get("/")
async def top_info():
    return {'about': 'mt-photos-ai-extra'}


@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {'result': 'pass'}


@app.post("/restart")
async def check_req(api_key: str = Depends(verify_header)):
    # cuda版本 OCR没有显存未释放问题，这边可以关闭重启
    return {'result': 'unsupported'}
    # restart_program()


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
        _result = await predict(ocr_model.ocr, img)
        result = trans_result(_result[0])
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
    python = sys.executable
    os.execl(python, python, *sys.argv)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
