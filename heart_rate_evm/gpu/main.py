import threading
from apply_evm import apply_evm
from camera import capture_frames
from collections import deque
from core.constants import BUFFER_SIZE


if __name__ == '__main__':

  buffer = deque(maxlen=BUFFER_SIZE)
  result = deque(maxlen=1)
  stop_event = threading.Event()

  camera_thread = threading.Thread(target=capture_frames, args=(buffer, result, stop_event,))
  evm_process = threading.Thread(target=apply_evm, args=(buffer, result, stop_event))

  camera_thread.start()
  evm_process.start()

  print("Finish")






