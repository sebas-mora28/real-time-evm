import cv2
import torch
from blazeface import BlazeFace
import time

# Initialize BlazeFace
print("Loading model...")
net = BlazeFace()
net.load_weights("blazeface.pth")  # download from official repo
net.load_anchors("anchors.npy")     # required anchor points
net.min_score = 0.5
print(torch.cuda.is_available())
net = net.to("cuda")
net.anchors = net.anchors.to("cuda") 
print("Finish to load")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().cuda()/255.0
    # Detect faces
    start = time.time()
    faces = net.predict_on_image(img_tensor[0])
    end= time.time()
    print("Time: ", (end - start)*1000)

    print(faces)

    # Draw boxes
    for x1, y1, x2, y2, conf in faces:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    
    cv2.imshow("BlazeFace Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

