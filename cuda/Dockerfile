FROM python:3.9-bullseye
USER root
# 镜像加速
COPY ./sources.list /etc/apt/sources.list
RUN apt update && \
    apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libfontconfig1 nvidia-cuda-dev nvidia-cuda-toolkit && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/log/*

WORKDIR /app

COPY requirements.txt .

# 安装依赖包
RUN python3 -m pip install paddlepaddle-gpu==2.5.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/cudnnin/stable.html && \
    pip3 install --no-cache-dir -r requirements.txt --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/
COPY server.py .

ENV API_AUTH_KEY=mt_photos_ai_extra
EXPOSE 8000

CMD [ "python3", "/app/server.py" ]
