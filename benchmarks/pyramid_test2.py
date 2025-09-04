import os
import cv2
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter

def apply_gaussian_pyr_levels(frames_gpu, levels=5, sigma=1):
    current = frames_gpu
    pyramid_levels = [current]
    for i in range(levels):
        current = gaussian_filter(current, sigma=(0, sigma, sigma), mode='reflect')
        current = current[:, ::2, ::2]
        pyramid_levels.append(current)
    return pyramid_levels

def apply_opencv_pyr_levels(frames_np, levels=5):
    current = frames_np
    pyramid_levels = [current]
    for i in range(levels):
        downsamples = []
        for img in current:
            down = cv2.pyrDown(img)
            downsamples.append(down)
        current = np.stack(downsamples, axis=0)
        pyramid_levels.append(current)
    return pyramid_levels

def save_pyramid_images(pyramid, base_path, prefix):
    os.makedirs(base_path, exist_ok=True)
    for level, frames in enumerate(pyramid):
        level_path = os.path.join(base_path, f"level_{level}")
        os.makedirs(level_path, exist_ok=True)
        for i, img in enumerate(frames):
            filename = os.path.join(level_path, f"{prefix}_frame_{i}.png")
            cv2.imwrite(filename, img)
            print(f"Guardado {filename}")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error abriendo la cámara")
        return

    buffer_size = 1
    green_buffer = []

    print("Capturando 5 frames...")
    for _ in range(buffer_size):
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame")
            break
        green = frame[:, :, 1]  # Canal verde
        green_buffer.append(green)

    cap.release()

    # Convertir a numpy array (N, H, W)
    green_np = np.stack(green_buffer, axis=0)

    # CuPy: a GPU
    green_gpu = cp.array(green_np, dtype=cp.float32)

    levels = 3
    sigma = 1

    print("Generando pirámide con Gaussian filter + downsample (GPU)...")
    gaussian_pyramid = apply_gaussian_pyr_levels(green_gpu, levels=levels, sigma=sigma)
    # Pasar a CPU uint8 y recortar
    gaussian_pyramid_cpu = [cp.asnumpy(frame).clip(0, 255).astype(np.uint8) for frame in gaussian_pyramid]

    print("Generando pirámide con cv2.pyrDown (CPU)...")
    opencv_pyramid = apply_opencv_pyr_levels(green_np, levels=levels)

    # Guardar imágenes
    save_pyramid_images(gaussian_pyramid_cpu, "gaussian_pyramid", "gaussian")
    save_pyramid_images(opencv_pyramid, "opencv_pyramid", "opencv")

    print("¡Listo!")

if __name__ == "__main__":
    main()
