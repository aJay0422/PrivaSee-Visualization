from mmpose.apis import MMPoseInferencer
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"


inferencer = MMPoseInferencer("human")

img_path = "./images/000000000785.jpg"

result_generator = inferencer(img_path, show=True)
result = next(result_generator)



