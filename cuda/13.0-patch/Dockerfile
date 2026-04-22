FROM mtphotos/mt-photos-ai:cuda-13.0

# 安装依赖包
RUN pip3 install --no-cache-dir RapidOCR==3.8.1 --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/
COPY ./models/ /opt/conda/lib/python3.11/site-packages/rapidocr/models/

ENV API_AUTH_KEY=mt_photos_ai_extra
ENV CLIP_MODEL=ViT-B-16

COPY server.py .

EXPOSE 8060

CMD [ "python3", "/app/server.py" ]
