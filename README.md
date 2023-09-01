# MT Photos AI识别相关任务独立部署项目

- 基于PaddleOCR实现的文本识别(OCR)接口

## 关于内存、显存占用

> 不断增长是因为分配的 shape 变大了，初始化创建 temporary tensor 等等会增长，等初始化之后跑起来，shape 不增大的话应该会稳定。
>
> 正常是过了最大的 shape 之后会保持稳定。 如果 shape 变小显存池尺寸也不会变小，目前机制会贪心增大但不会降低。
>
> 如果你想控制最大显存变小一点，可以减小输入图像的尺度：将params.py中的参数det_max_side_len从默认的960改成680或者更小尺度。

原文地址：

https://github.com/PaddlePaddle/PaddleOCR/issues/489

## 目录说明

- common：基于paddleocr库自带的api来进行识别任务
- openvino：基于OpenVINO，调用PaddleOCR模型来进行识别任务

在intel cpu上运行时OpenVINO版本会快很多,差不多有10倍的提升；

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

## docker打包 & 运行

### OpenVINO版本

```bash
cd openvino
docker build  . -t mt-photos-ai:latest
```

```bash
docker run -i -p 8000:8000 -e API_AUTH_KEY=mt_photos_ai_extra_secret --name mt-photos-ai --restart="unless-stopped" mt-photos-ai:latest
```

mt_photos_ai_extra_secret 为验证api请求的api_key,请替换

### common版本

```bash
cd common
docker build  . -t mt-photos-ai:cpu-latest
```

```bash
docker run -i -p 8000:8000 -e API_AUTH_KEY=mt_photos_ai_extra_secret --name mt-photos-ai --restart="unless-stopped" mt-photos-ai:cpu-latest
```

mt_photos_ai_extra_secret 为验证api请求的api_key,请替换

## 本地运行或者打包openvino文件夹下需要额外下载的文件

下载以下2个文件,然后解压放到model文件夹下

- https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar
- https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar


```bash
# tree ./openvino 
|-- model
|   |-- ch_PP-OCRv4_det_infer
|   |   |-- inference.pdiparams
|   |   |-- inference.pdiparams.info
|   |   `-- inference.pdmodel
|   |-- ch_PP-OCRv4_rec_infer
|   |   |-- inference.pdiparams
|   |   |-- inference.pdiparams.info
|   |   `-- inference.pdmodel
|   `-- ppocr_keys_v1.txt
|-- pre_post_processing.py
|-- requirements.txt
|-- server.py
`-- vino.py
```

### 本地运行

 - 安装python3，推荐版本3.8；
 - 打开 common 或  openvino文件夹
 - pip install -r requirements.txt
 - openvino文件夹下需要下model下的相关模型文件，common不用下载
 - python3 server.py
