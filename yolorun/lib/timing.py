from contextlib import contextmanager
from time import time


@contextmanager
def timing(description: str) -> None:
    start = time()
    yield
    elapsed_time = time() - start

    print(f"{description}: {1000*elapsed_time:.1f}ms")



class Counter:
    def __init__(self, description="counter"):
        self._start = time()
        self.counter = 0
        self.mean = .0
        self.description = description
    
    # @classmethod
    # @contextmanager
    # def start(cls) -> None:
    #     #self.start = time()
    #     yield

    # #def end(self):

    def __str__(self):
        return f"{self.description} mean={1000*self.mean:.1f}ms counter={self.counter}"

    def start(self):
        self._start = time()

    def end(self):
        elapsed = time() - self._start
        self.counter += 1
        self.mean = self.mean*(self.counter-1)/self.counter + elapsed/self.counter
