import cv2
from cupyx.scipy.ndimage import gaussian_filter

def get_gaussian_pyramid(frame_green_channel, levels):
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame_green_channel)
    current = gpu_frame
    for level in range(levels):
      current = cv2.cuda.pyrDown(current)
    current_cpu = current.download()
    return current_cpu


def get_gaussian_pyramid_gpu(frames, levels):
    current = frames
    for i in range(levels):
        blurred = gaussian_filter(current, sigma=(0, 1, 1))  # adjust sigma
        current = blurred[:, ::2, ::2]  # downsample by factor of 2
    return current