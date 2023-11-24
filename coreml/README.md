# How to use

1. Use [pytorch2coreml](https://github.com/manymuch/Chinese-CLIP/blob/master/cn_clip/deploy/pytorch_to_coreml.py) converter to get two mlpackage (text and image)
2. prepare the two mlpackage folder under ``coreml/utils/``
3. rename .env.example to .env and fill ``API_AUTH_KEY``
4. Run ``python3 server.py``
    