# cuda-12.9  tag镜像补丁包

原始镜像 RapidOCR 2.0.6版本太低，升级RapidOCR版本

### 打包docker镜像

```bash
docker build  . -t mt-photos-ai:cuda-12.9-sp1
docker run --gpus all -i -p 8060:8060 -e API_AUTH_KEY=mt_photos_ai_extra --name mt-photos-ai-cuda mt-photos-ai:cuda-12.9-sp1

```
