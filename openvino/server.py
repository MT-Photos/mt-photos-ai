from dotenv import load_dotenv
import os
import sys
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
import uvicorn
import numpy as np
import cv2
import asyncio
from pydantic import BaseModel
from rapidocr_openvino import RapidOCR

import utils.clip as clip

on_linux = sys.platform.startswith('linux')

load_dotenv()
app = FastAPI()
api_auth_key = os.getenv("API_AUTH_KEY")

inactive_task = None
rapid_ocr = None
clip_img_model = None
clip_txt_model = None

class ClipTxtRequest(BaseModel):
    text: str

def load_ocr_model():
    global rapid_ocr
    if rapid_ocr is None:
        rapid_ocr = RapidOCR()

def load_clip_img_model():
    global clip_img_model
    if clip_img_model is None:
        clip_img_model = clip.load_img_model()

def load_clip_txt_model():
    global clip_txt_model
    if clip_txt_model is None:
        clip_txt_model = clip.load_txt_model()

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
        texts.append(res_i[1])
        scores.append(res_i[2][:4])
    return {'texts': texts, 'scores': scores, 'boxes': boxes}


@app.get("/")
async def top_info():
    return {'about': 'mt-photos-ai-extra'}


@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {'result': 'pass'}


@app.post("/restart")
async def check_req(api_key: str = Depends(verify_header)):
    # 已知的问题：使用openVINO进行OCR识别，存在内存泄露，进程的内存占用会一直增加；
    # 客户端可调用，触发重启进程来释放内存
    restart_program()

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
        _result = rapid_ocr(img)
        result = trans_result(_result[0])
        del img
        del _result
        return {'result': result}
    except Exception as e:
        print(e)
        return {'result': [], 'msg': str(e)}


@app.post("/clip/img")
async def clip_process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    load_clip_img_model()
    image_bytes = await file.read()
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = await predict(clip.process_image, img, clip_img_model)
        return {'result': ["{:.16f}".format(vec) for vec in result]}
    except Exception as e:
        print(e)
        return {'result': [], 'msg': str(e)}

@app.post("/clip/txt")
async def clip_process_txt(request:ClipTxtRequest, api_key: str = Depends(verify_header)):
    load_clip_txt_model()
    text = request.text
    result = await predict(clip.process_txt, text, clip_txt_model)
    return {'result': ["{:.16f}".format(vec) for vec in result]}

async def predict(predict_func, inputs,model):
    return await asyncio.get_running_loop().run_in_executor(None, predict_func, inputs,model)

def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
