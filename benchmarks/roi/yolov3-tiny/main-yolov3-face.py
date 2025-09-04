import cv2
import time

# Configuración del modelo YOLOv4-tiny para detección de caras
model_cfg = "yolov3-face.cfg"
model_weights = "yolov3-wider_16000.weights"

# Cargar la red con OpenCV (Darknet backend)
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)   # Usar GPU (Jetson Nano)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

# Preparar cámara
cap = cv2.VideoCapture(0)  # Cámara USB / CSI
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Obtener nombres de capas de salida
layer_names = net.getLayerNames()
print(net.getUnconnectedOutLayers())
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Crear blob para red neuronal
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    detections = net.forward(output_layers)
    end = time.time()
    print("Time: ", (end -start)*1000)

    # Procesar detecciones
    boxes = []
    confidences = []
    for output in detections:
        for detection in output:
            scores = detection[5:]  # Confianza desde la clase
            confidence = scores[0]  # Solo hay una clase: "face"
            if confidence > 0.5:
                box = detection[0:4] * [w, h, w, h]
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    # Non-Maximum Suppression para evitar cajas repetidas
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        i = i[0] if isinstance(i, (list, tuple)) else i
        (x, y, w_box, h_box) = boxes[0]
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv2.putText(frame, f"Face",
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    fps = 1 / (end - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLOv4-tiny Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
