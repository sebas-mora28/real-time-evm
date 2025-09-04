import cv2
import time

face_cascade_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"


face_cascade_cpu = cv2.CascadeClassifier(face_cascade_path)
#face_cascade_gpu = cv2.cuda_CascadeClassifier.create("./haarcascade_frontalface_default.xml")


cap = cv2.VideoCapture(0)

i = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # --------------- CPU -------------------------
    start_cpu = time.time()
    face_cascade_cpu.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    end_cpu = time.time()


    # ------------- GPU ---------------------------
    #gpu_mat = cv2.cuda_GpuMat()
    #gpu_mat.upload(gray)
    start_gpu = time.time()
    #face_cascade_gpu.detectMultiScale(gpu_mat)
    #gpu_mat.download()
    end_gpu = time.time()


    print(f"CPU: {(end_cpu - start_cpu)*1000:.2f} ms | GPU {(end_gpu - start_gpu)*1000:.2f} ms")

    i += 1

    if(i == 5):
        break
    
    if (cv2.waitKey(1) & 0xFF == ord("q")):
        break

cap.release()
cv2.destroyAllWindows()
    
