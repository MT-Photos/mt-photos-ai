from dotenv import load_dotenv
import os
import sys
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
import uvicorn
import numpy as np
import cv2
import vino as v
import asyncio
import copy

on_linux = sys.platform.startswith('linux')

load_dotenv()
app = FastAPI()
app.state.ocr_model = None
api_auth_key = os.getenv("API_AUTH_KEY")

ocr_model_loading = False
inactive_task = None


async def check_inactive():
    # 超过5分钟没请求,释放模型占用的内存
    await asyncio.sleep(300)
    await release_ocr_model()


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


@app.on_event("startup")
async def startup_event() -> None:
    load_model()


@app.get("/")
async def top_info():
    return {'about': 'mt-photos-ai-extra'}


@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {'result': 'pass'}


@app.post("/restart")
async def check_req(api_key: str = Depends(verify_header)):
    if on_linux:
        # 未找到释放内存的方法，重启进程
        await release_ocr_model()
    else:
        return {'result': 'unsupported'}


@app.post("/ocr")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    if ocr_model_loading is True:
        return {'code': "wait_load_model"}

    if app.state.ocr_model is None:
        load_model()

    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    _result = v.run_paddle_ocr(img, app.state.ocr_model)
    # 模型推理未实现异步，无法使用多线程
    # result = await predict(v.run_paddle_ocr, img, app.state.ocr_model)
    result = copy.deepcopy(_result)
    del img
    del _result
    return {'result': result}


async def predict(predict_func, inputs, model):
    return await asyncio.get_running_loop().run_in_executor(None, predict_func, inputs, model)


def load_model():
    global ocr_model_loading
    print('load model')
    ocr_model_loading = True
    app.state.ocr_model = v.load_model()
    ocr_model_loading = False


async def release_ocr_model():
    # 轻量模型内存会涨到2G，通用模型涨到5G。
    # 不断增长的内存占用是因为分配的 shape 变大了，初始化创建 temporary tensor 等等会增长，
    # 等初始化之后跑起来，shape 不增大的话应该会稳定。
    # 正常是过了最大的 shape 之后会保持稳定。
    # 如果 shape 变小显存池尺寸也不会变小，目前机制会贪心增大但不会降低。
    # https://github.com/PaddlePaddle/PaddleOCR/issues/489

    if on_linux:
        # linux未找到释放内存的方法，重启进程
        restart_program()
        return

    # 部分情况下windows可以释放内存，有时无法释放，原因未知
    del app.state.ocr_model
    await asyncio.sleep(1)
    app.state.ocr_model = None


def restart_program():
    # 仅支持linux，Windows需要使用subprocess模块来执行一个新的Python进程
    print('restart')
    python = sys.executable
    os.execl(python, python, *sys.argv)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
