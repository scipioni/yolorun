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
        self._data = {}

    # @classmethod
    # @contextmanager
    # def start(cls) -> None:
    #     #self.start = time()
    #     yield

    # #def end(self):

    def __str__(self):
        response = []
        for key in self._data:
            response.append(f"{key} mean={1000*self._data[key]['mean']:.1f}ms counter={self._data[key]['counter']}")
        return "\n".join(response)

    @contextmanager
    def __call__(self, key="generic"):
        if not key in self._data:
            self._data[key] = {"start":0, "mean":0, "counter":0}
        self._data[key]["start"] = time()
        yield
        elapsed = time() - self._data[key]["start"]
        self._data[key]["counter"] += 1
        self._data[key]["mean"] = self._data[key]["mean"]*(self._data[key]["counter"]-1)/self._data[key]["counter"] + elapsed/self._data[key]["counter"]
