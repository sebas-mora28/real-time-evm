import cv2
import time

# Paths to model files
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"

# Load the network
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Enable CUDA backend (Jetson Nano GPU)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

# Load an image
cap = cv2.VideoCapture(0)  # Cámara USB / CSI
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Prepare input blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                             1.0, (300, 300), 
                             (104.0, 177.0, 123.0))

    # Warm-up run
    net.setInput(blob)
    net.forward()

    # Measure inference time
    start = time.time()
    net.setInput(blob)
    detections = net.forward()
    end = time.time()

    print(f"Inference time (CUDA): {(end - start) * 1000:.2f} ms")

    # Process detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            #print(f"Detected ROI: {(x1, y1, x2, y2)} conf={confidence:.2f}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    fps = 1 / (end -start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show result
    cv2.imshow(f"Face Detection (CUDA)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    
cv2.waitKey(0)
cv2.destroyAllWindows()

