from time import perf_counter

class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = perf_counter() - self.start