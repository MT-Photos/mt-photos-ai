# MT Photos AI识别相关任务独立部署项目

- 基于PaddleOCR实现的文本识别(OCR)接口

## 关于内存、显存占用

> 轻量模型会涨到2G，通用模型涨到5G。Docker镜像使用的是轻量模型。
> 
> 内存不断增长是因为分配的 shape 变大了，初始化创建 temporary tensor 等等会增长，等初始化之后跑起来，shape 不增大的话应该会稳定。
>
> 正常是过了最大的 shape 之后会保持稳定。 如果 shape 变小显存池尺寸也不会变小，目前机制会贪心增大但不会降低。
>
> 如果你想控制最大显存变小一点，可以减小输入图像的尺度：将params.py中的参数det_max_side_len从默认的960改成680或者更小尺度。

原文地址：

https://github.com/PaddlePaddle/PaddleOCR/issues/489

## 目录说明

- openvino：基于rapidocr_openvino库，进行识别任务。**适用于Intel CPU运行**
- onnx：基于rapidocr_onnxruntime库，进行识别任务。**适用于AMD CPU运行**
- cuda：基于paddleocr官方库，进行识别任务。**适用于支持CUDA的显卡运行**

> 在Intel cpu上运行时OpenVINO版本会快很多；

>RapidOCR更多配置可参考官方仓库 https://github.com/RapidAI/RapidOCR

## 镜像说明

DockerHub镜像仓库地址：
https://hub.docker.com/r/mtphotos/mt-photos-ai

镜像Tags说明：

- latest：基于openvino文件夹打包生成，推荐**Intel CPU**机型安装这个镜像
- onnx:基于onnx文件夹打包生产，推荐**AMD CPU**机型安装这个镜像

由于cuda版本镜像包含的驱动等相关文件较多，未打包镜像，有需要可以自行打包。


### 打包docker镜像

```bash
cd cuda
docker build  . -t mt-photos-ai:cuda-latest
```
`cuda`为文件夹，可根据需要替换为`onnx`、`openvino`

### 运行docker容器

```bash
docker run -i -p 8000:8000 -e API_AUTH_KEY=mt_photos_ai_extra_secret --name mt-photos-ai --restart="unless-stopped" mt-photos-ai:cuda-latest
```
`cuda-latest`可以替换为`latest`、`cpu-latest`


### 下载源码本地运行

- 安装python **3.8版本**，实测再更高版本上无法运行
- 根据硬件环境选择cuda、onnx或openvino文件夹
- 在选择文件夹下执行`pip install -r requirements.txt`
- 复制`.env.example`生成`.env`文件，然后修改`.env`文件内的API_AUTH_KEY
- 执行 `python server.py` ，启动服务

> python安装包地址： https://www.python.org/downloads/release/python-3810/
> 
> API_AUTH_KEY为MT Photos填写api_key需要输入的值

看到以下日志，则说明服务已经启动成功
```bash
INFO:     Started server process [3024]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```


## API

### /check

检测服务是否可用，及api-key是否正确

```bash
curl --location --request POST 'http://127.0.0.1:8000/check' \
--header 'api-key: api_key'
```

**response:**

```json
{
  "result": "pass"
}
```

### /ocr

```bash
curl --location --request POST 'http://127.0.0.1:8000/ocr' \
--header 'api-key: api_key' \
--form 'file=@"/path_to_file/test.jpg"'
```

**response:**

- result.texts : 识别到的文本列表
- result.scores : 为识别到的文本对应的置信度分数，1为100%
- result.boxes : 识别到的文本位置，x,y为左上角坐标，width,height为框的宽高

```json
{
  "result": {
    "texts": [
      "识别到的文本1",
      "识别到的文本2"
    ],
    "scores": [
      "0.98",
      "0.97"
    ],
    "boxes": [
      {
        "x": "4.0",
        "y": "7.0",
        "width": "283.0",
        "height": "21.0"
      },
      {
        "x": "7.0",
        "y": "34.0",
        "width": "157.0",
        "height": "23.0"
      }
    ]
  }
}
```

### /restart

通过重启进程来释放内存

```bash
curl --location --request POST 'http://127.0.0.1:8000/restart' \
--header 'api-key: api_key'
```

**response:**

请求中断,没有返回，因为服务重启了
