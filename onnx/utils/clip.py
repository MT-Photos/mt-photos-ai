import os
import sys
import numpy as np
from PIL import Image
from typing import Union, List
import onnxruntime

current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_folder)
import bert_tokenizer as bert


def join_path(folder_path, file_name):
    return os.path.join(folder_path, file_name)


model_folder_path = current_folder
img_onnx_model_path = join_path(model_folder_path, "vit-b-16.img.fp32.onnx")
txt_onnx_model_path = join_path(model_folder_path, "vit-b-16.txt.fp32.onnx")

IMG_SIZE = 224

_tokenizer = bert.FullTokenizer()
mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def single_image_transform(image, image_size):
    image = Image.fromarray(np.uint8(image)).resize((image_size, image_size), Image.BICUBIC)
    image = np.array(Image.fromarray(np.uint8(image)).convert('RGB'))
    image = np.array(image, dtype=np.float32) / 255.0
    image = (image - mean) / std
    return image.astype(np.float32)


def image_processor(image_batch, image_size=224):
    transformed_batch = [single_image_transform(img, image_size) for img in image_batch]
    transformed_batch = np.array(transformed_batch, dtype=np.float32)  # Shape would be (N, H, W, C)

    # Reorder dimensions to (N, C, H, W)
    transformed_batch = np.transpose(transformed_batch, (0, 3, 1, 2))

    return transformed_batch


def tokenize_numpy(texts: Union[str, List[str]], context_length: int = 52) -> np.ndarray:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all baseline models use 52 as the context length
    Returns
    -------
    A two-dimensional numpy array containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append([_tokenizer.vocab['[CLS]']] + _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))[
                                                        :context_length - 2] + [_tokenizer.vocab['[SEP]']])

    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = np.array(tokens)

    return result


def load_img_model():
    img_sess_options = onnxruntime.SessionOptions()
    img_run_options = onnxruntime.RunOptions()
    img_run_options.log_severity_level = 2
    return onnxruntime.InferenceSession(img_onnx_model_path,
                                        sess_options=img_sess_options,
                                        providers=["CPUExecutionProvider"])


def process_image(img, img_model):
    inputs = image_processor([img], image_size=IMG_SIZE)
    input = inputs[0:1, :, :, :]
    image_feature = img_model.run(["unnorm_image_features"], {"image": input})[0].tolist()[0]
    return image_feature


def load_txt_model():
    txt_sess_options = onnxruntime.SessionOptions()
    txt_run_options = onnxruntime.RunOptions()
    txt_run_options.log_severity_level = 2
    return onnxruntime.InferenceSession(txt_onnx_model_path,
                                        sess_options=txt_sess_options,
                                        providers=["CPUExecutionProvider"])


def process_txt(txt, text_model):
    input = tokenize_numpy([txt], 52)
    embeddings = text_model.run(["unnorm_text_features"], {"text": input})[0].tolist()[0]
    return embeddings
