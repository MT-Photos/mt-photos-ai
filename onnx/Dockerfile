FROM python:3.8.10-buster
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
COPY server.py .
COPY ./utils ./utils

ENV API_AUTH_KEY=mt_photos_ai_extra
EXPOSE 8000

CMD [ "python3", "server.py" ]
