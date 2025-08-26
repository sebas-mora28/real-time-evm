import cv2
import numpy as np
import time

try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    frame = frame[:, :, 1]
    # --- NumPy FFT ---
    t0 = time.time()
    np_fft = np.fft.fft2(frame)
    np_ms = (time.time() - t0) * 1000

    # --- CuPy FFT ---
    cp_ms = None
    if has_cupy:
        t0 = time.time()
        d_gray = cp.asarray(frame)
        d_fft = cp.fft.fft2(d_gray)
        cp.cuda.Stream.null.synchronize()   # asegurar tiempo real
        cp_ms = (time.time() - t0) * 1000

    # Mostrar tiempos en consola
    if cp_ms:
        print(f"NumPy: {np_ms:.2f} ms | CuPy: {cp_ms:.2f} ms")
    else:
        print(f"NumPy: {np_ms:.2f} ms | CuPy no disponible")

    i += 1

    if(i == 50):
        break

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
