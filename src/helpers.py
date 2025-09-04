import cv2 
import numpy as np


heart_rate = None
roi = None 

def show_frame(frame, result):
    global heart_rate, roi

    if(len(result) != 0):
        values = result.pop()
        heart_rate = values['heart_rate']
        roi = values['roi']

    cv2.putText(frame, 
                f"BPM: {heart_rate}", 
                org=(50, 50), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, 
                color=(0, 255, 0),
                thickness=2, 
                lineType=cv2.LINE_AA)
    
    if(roi is not None):
        (x, y, w, h) = roi 
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    

    cv2.imshow("Heart Rate Monitor", frame)

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

    video.release()

    return np.asarray(frames), fps


def save_video(frames, filename, fps=30):
    """
    Saves a video from a NumPy array of shape (N, H, W, 3).
    
    Args:
        frames: NumPy array of shape (num_frames, height, width, 3)
        filename: Output video filename (e.g., "output.avi" or "output.mp4")
        fps: Frames per second (default is 30)
    """
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if filename.endswith(".mp4") else cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for i in range(len(frames)):

        # Convert from float32 to uint8 if needed
        frame = frames[i]
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        
        out.write(frame)
    
    out.release()
    print(f"Video saved to {filename}")
