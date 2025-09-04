import cv2
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
import numpy as np
import os

# Crear carpetas para el nivel 2
os.makedirs("pyrdown_opencv/lvl2", exist_ok=True)
os.makedirs("pyrdown_cupy/lvl2", exist_ok=True)

def pyrdown_cupy(image, sigma=0.6):
    img_gpu = cp.array(image, dtype=cp.float32)
    channels = []
    for c in range(img_gpu.shape[2]):
        blurred = gaussian_filter(img_gpu[:, :, c], sigma=sigma)
        channels.append(blurred)
    blurred_gpu = cp.stack(channels, axis=2)
    downsampled_gpu = blurred_gpu[::2, ::2, :]
    return cp.asnumpy(cp.clip(downsampled_gpu, 0, 255)).astype(np.uint8)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_id = 0

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("Presiona 'q' para salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame.")
            break

        print(frame.shape)

        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Nivel 1 OpenCV
        pyr_opencv_lvl1 = cv2.cuda.pyrDown(gpu_frame)
        # Nivel 2 OpenCV
        pyr_opencv_lvl2 = cv2.cuda.pyrDown(pyr_opencv_lvl1)

        # Nivel 1 CuPy
        pyr_cupy_lvl1 = pyrdown_cupy(frame, sigma=0.6)
        # Nivel 2 CuPy
        pyr_cupy_lvl2 = pyrdown_cupy(pyr_cupy_lvl1, sigma=0.6)

        pyr_opencv_lvl2 = pyr_opencv_lvl2.download()

        # Calcular diferencia absoluta
        diff = cv2.absdiff(pyr_opencv_lvl2, pyr_cupy_lvl2)
        diff_sum = np.sum(diff)

        print(f"Frame {frame_id}: Suma diferencia absoluta = {diff_sum}")

        # Mostrar solo el último nivel
        cv2.imshow("OpenCV pyrDown lvl2", pyr_opencv_lvl2)
        cv2.imshow("CuPy pyrDown lvl2", pyr_cupy_lvl2)
        cv2.imshow("Diferencia absoluta", diff)

        # Guardar imágenes
        filename = f"frame_{frame_id:04d}.png"
        cv2.imwrite(f"pyrdown_opencv/lvl2/{filename}", pyr_opencv_lvl2)
        cv2.imwrite(f"pyrdown_cupy/lvl2/{filename}", pyr_cupy_lvl2)

        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Proceso terminado.")

if __name__ == "__main__":
    main()