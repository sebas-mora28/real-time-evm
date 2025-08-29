import cv2

def get_gaussian_pyramid(frame_green_channel, levels):
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame_green_channel)
    current = gpu_frame
    for level in range(levels):
      current = cv2.cuda.pyrDown(current)
    current_cpu = current.download()
    return current_cpu
