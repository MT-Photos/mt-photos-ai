## How to convert a python model to CoreML model

1. Heading to (chinese-clip)[https://github.com/OFA-Sys/Chinese-CLIP] and follow the instructions to install the required packages.
2. Replace the `cn_clip/deploy/pytorch_to_coreml.py` with the `pytorch_to_coreml.py` in this repository.
3. Follow the instructions in the chinese-clip repository to convert the model to CoreML model.

__note__: Please ues torch 2.2.0, 2.3.0 will lead to an error when loading the model in CoreML. 

## Data Structure

### Input data structure:

| Part       | Description                    | Data Type | Length                  |
|------------|--------------------------------|-----------|-------------------------|
| Message ID | MD5 hash of the image data     | String    | 32 bytes (fixed length) |
| Image Data | Binary data of the image       | Byte Data | Variable length         |

### Output data structure:
```json
{
	"id": "d41d8cd98f00b204e9800998ecf8427e",
	"embedding": [0.1, 0.2, 0.3, ..., 0.9],
	"ocr_result": [
		{
			"text": "text1",
			"boundingBox": {
				"height": 41.99999999999998,
				"width": 282,
				"x": 26.000002014285684,
				"y": 424.0000001
			},
			"confidence": 0.5
		},
		{
			"text": "text2",
			"boundingBox": {
				"height": 27.999999999999968,
				"x": 23.999998916666684,
				"width": 150.00000000000003,
				"y": 394.00000023333337
			},
			"confidence": 1
		}
	]
}
```
