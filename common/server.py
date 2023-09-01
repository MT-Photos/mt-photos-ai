from dotenv import load_dotenv
import os
import sys
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
import uvicorn
import numpy as np
import cv2
import asyncio
from paddleocr import PaddleOCR

on_linux = sys.platform.startswith('linux')

load_dotenv()
app = FastAPI()
api_auth_key = os.getenv("API_AUTH_KEY")

inactive_task = None

ocr_model = PaddleOCR(
    show_log=False,
    use_angle_cls=True,
    lang="ch",
    ocr_version="PP-OCRv4",
    use_gpu=True
)
# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/whl.md

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
    # 轻量模型内存会涨到2G，通用模型涨到5G。
    # 不断增长的内存占用是因为分配的 shape 变大了，初始化创建 tempory tensor 等等会增长，
    # 等初始化之后跑起来，shape 不增大的话应该会稳定。
    # 正常是过了最大的 shape 之后会保持稳定。
    # 如果 shape 变小显存池尺寸也不会变小，目前机制会贪心增大但不会降低。

    if on_linux:
        # 容器内，无法有效释放内存，重启进程
        restart_program()
    else:
        return {'result': 'unsupported'}


@app.post("/ocr")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    _result = await predict(ocr_model.ocr, img)
    result = trans_result(_result[0])
    del img
    del _result
    return {'result': result}


async def predict(predict_func, inputs):
    return await asyncio.get_running_loop().run_in_executor(None, predict_func, inputs)


def restart_program():
    # 仅支持linux，Windows需要使用subprocess模块来执行一个新的Python进程
    print('restart')
    if on_linux:
        # 容器内，无法有效释放内存，重启进程
        python = sys.executable
        os.execl(python, python, *sys.argv)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
