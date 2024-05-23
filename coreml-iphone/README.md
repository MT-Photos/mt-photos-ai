## How to convert a python model to CoreML model

1. Heading to (chinese-clip)[https://github.com/OFA-Sys/Chinese-CLIP] and follow the instructions to install the required packages.
2. Replace the `cn_clip/deploy/pytorch_to_coreml.py` with the `pytorch_to_coreml.py` in this repository.
3. Follow the instructions in the chinese-clip repository to convert the model to CoreML model.

__note__: Please ues torch 2.2.0, 2.3.0 will lead to an error when loading the model in CoreML. 
