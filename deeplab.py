import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from framemask import overlay


model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)

model.eval()

image = cv2.imread("./images/table1.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

device = torch.device("cuda"
                      if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
device = torch.device("cpu")

input_batch = input_batch.to(device)
model.to(device)


with torch.no_grad():
    output = model(input_batch)["out"][0]

output_predictions = output.detach().cpu().numpy().argmax(0)

target_point = np.array([600, 300])
target_cls = output_predictions[target_point[1], target_point[0]]
mask_bool = np.zeros_like(output_predictions)
mask_bool[output_predictions == target_cls] = int(1)

plt.imshow(mask_bool)

r = overlay(image, np.array([mask_bool]))

cv2.imshow("orig image", mask_bool)
cv2.waitKey(5000)