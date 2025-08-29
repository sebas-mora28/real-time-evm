import threading
import multiprocessing
from buffer import Buffer
from apply_evm import apply_evm
from camera import capture_frames
from collections import deque
from constants import BUFFER_SIZE


if __name__ == '__main__':

  buffer = deque(maxlen=BUFFER_SIZE)
  stop_event = threading.Event()

  camera_thread = threading.Thread(target=capture_frames, args=(buffer, roi, stop_event,))
  evm_process = threading.Thread(target=apply_evm, args=(buffer, stop_event))

  camera_thread.start()
  evm_process.start()







