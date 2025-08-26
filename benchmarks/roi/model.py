import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
import cv2

# Load pretrained SSD model (COCO)
print("LLega")
model = ssd300_vgg16(pretrained=True)
model.eval()
print("Pasa")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
model.to(device)

from PIL import Image
from torchvision.transforms import functional as F

print("Commienza a descargar la imagen")
img_cv2 = cv2.imread("../data/face.png")
img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)  # Convert to RGB for PyTorch
img_rgb = cv2.resize(img_rgb, (200, 200))
img_pil = Image.fromarray(img_rgb)
img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(device)


print("Iniciating model")
with torch.no_grad():
    outputs = model(img_tensor)
print("Model finished")

# outputs[0] contains boxes, labels, scores
boxes = outputs[0]['boxes']
labels = outputs[0]['labels']
scores = outputs[0]['scores']

# Filter by confidence
threshold = 0.5
boxes = boxes[scores > threshold]
labels = labels[scores > threshold]
scores = scores[scores > threshold]

threshold = 0.5
boxes = boxes[scores > threshold].cpu().numpy()

print(boxes)

for box in boxes:
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle

# Save output image
cv2.imwrite("face_detected.jpg", img_cv2)
print("Saved image with rectangles as face_detected.jpg")