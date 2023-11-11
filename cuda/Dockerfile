FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
USER root
# 镜像加速
COPY ./sources.list /etc/apt/sources.list
RUN apt update && \
    apt install -y wget libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libfontconfig  python3 pip && \
    wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.20_amd64.deb && \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2.20_amd64.deb && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/log/*

WORKDIR /app

COPY requirements.txt .

# 安装依赖包
RUN python3 -m pip install paddlepaddle-gpu==2.5.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/cudnnin/stable.html && \
    pip3 install --no-cache-dir -r requirements.txt --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/
COPY server.py .

ENV API_AUTH_KEY=mt_photos_ai_extra
ENV CLIP_MODEL=ViT-B-16
EXPOSE 8000

CMD [ "python3", "/app/server.py" ]
