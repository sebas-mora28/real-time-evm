import cv2
import numpy as np

def load_video(path):
    frames = []
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)

    while video.isOpened():
        success, frame = video.read()

        if not success:
            break

        rgb_frame = frame[:, :, ::-1] 
        frames.append(rgb_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    return np.asarray(frames), fps

