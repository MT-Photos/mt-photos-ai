# cuda-13.0  tag镜像补丁包

原始镜像 RapidOCR 2.0.6版本太低，升级RapidOCR版本

### 打包docker镜像

```bash
docker build  . -t mt-photos-ai:cuda-13.0-sp1
docker run --gpus all -i -p 8060:8060 -e API_AUTH_KEY=mt_photos_ai_extra --name mt-photos-ai-cuda mt-photos-ai:cuda-13.0-sp1

docker tag  mt-photos-ai:cuda-13.0-sp1  mtphotos/mt-photos-ai:cuda-13.0-sp1
docker push mtphotos/mt-photos-ai:cuda-13.0-sp1

docker tag  mt-photos-ai:cuda-13.0-sp1  mtphotos/mt-photos-ai:cuda-13.0
docker push mtphotos/mt-photos-ai:cuda-13.0

docker tag  mt-photos-ai:cuda-13.0-sp1  registry.cn-hangzhou.aliyuncs.com/mtphotos/mt-photos-ai:cuda-13.0
docker push registry.cn-hangzhou.aliyuncs.com/mtphotos/mt-photos-ai:cuda-13.0


```
