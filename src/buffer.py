from multiprocessing import Queue
import queue

class Buffer:
    def __init__(self, maxsize):
        self.queue = Queue(maxsize=maxsize)
        self.maxsize = maxsize

    def put(self, item):
        if(self.queue.full()):
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put(item)

    def get(self):
        return self.queue.get()
    
    def full(self):
        return self.queue.full()
    
