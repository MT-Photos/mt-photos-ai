import os
import sys
import numpy as np
from PIL import Image, ImageFile
from typing import Union, List
import coremltools
ImageFile.LOAD_TRUNCATED_IMAGES = True

current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_folder)
import bert_tokenizer as bert


def join_path(folder_path, file_name):
    return os.path.join(folder_path, file_name)


model_folder_path = current_folder
img_coreml_model_path = join_path(model_folder_path, "clip_cn_vit-b-16.image.mlpackage/Data/com.apple.CoreML/model.mlmodel")
txt_coreml_model_path = join_path(model_folder_path, "clip_cn_vit-b-16.text.mlpackage/Data/com.apple.CoreML/model.mlmodel")

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
    model = coremltools.models.MLModel(img_coreml_model_path)
    input_description = model.get_spec().description.input
    # Extract the shape for the input named "image"
    for input_feature in input_description:
        if input_feature.name == "image" and input_feature.type.HasField('multiArrayType'):
            shape = input_feature.type.multiArrayType.shape
            # Assuming the shape represents [batch, channels, height, width]
            global image_width
            image_width = shape[-1]
    return model


def process_image(img, img_model):
    inputs = image_processor([img], image_size=image_width)
    input = inputs[0:1, :, :, :]
    input_data = {'image': input}
    image_feature = img_model.predict(input_data)["var_1350"][0].tolist()
    return image_feature


def load_txt_model():
    model = coremltools.models.MLModel(txt_coreml_model_path)
    return model


def process_txt(txt, text_model):
    input = tokenize_numpy([txt], 52).astype(np.int32)
    input_data = {'text': input}
    embeddings = text_model.predict(input_data)["var_1113"][0].tolist()
    return embeddings
