FROM python:3.11.8-bookworm
USER root
# 镜像加速
COPY ./sources.list /etc/apt/sources.list
RUN apt update && \
    apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libfontconfig1 && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/log/*

WORKDIR /app

COPY requirements.txt .

# 安装依赖包
RUN pip3 install --no-cache-dir -r requirements.txt --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/

COPY ./utils/vit-b-16.img.fp32.onnx ./utils/vit-b-16.img.fp32.onnx
COPY ./utils/vit-b-16.txt.fp32.onnx ./utils/vit-b-16.txt.fp32.onnx
COPY ./bert_tokenizer.py ./bert_tokenizer.py
COPY ./vocab.txt ./vocab.txt
COPY ./clip.py ./clip.py
COPY ./server.py ./server.py

ENV API_AUTH_KEY=mt_photos_ai_extra
EXPOSE 8060

CMD [ "python3", "server.py" ]
