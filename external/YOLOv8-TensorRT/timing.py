from contextlib import contextmanager
from time import time
import logging

logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
        level="DEBUG",
    )

log = logging.getLogger(__name__)

global_counters = {}


@contextmanager
def timing(description: str, level=1, count=10) -> None:
    global global_counters
    start = time()
    yield
    elapsed_time = 1000 * (time() - start)

    if description not in global_counters:
        global_counters[description] = {
            "n": 0,
            "mean": 0.0,
            "current": 0.0,
            "max": 0.0,
            "min": 999.0,
            "level": level,
        }

    counters = global_counters[description]
    n = counters["n"]

    counters["current"] = elapsed_time
    n += 1
    counters["n"] = n

    if n <= 1: # il primo ciclo può essere lento, quindi lo escludiamo
        return

    counters["mean"] = (elapsed_time + counters["mean"] * (n - 1)) / n
    counters["max"] = max(elapsed_time, counters["max"])
    counters["min"] = min(elapsed_time, counters["min"])

    if n % count == 0:
        log.debug(
            "%s%s timing: n=%d current=%.1fms min=%.1fms max=%.1fms mean=%.1fms"
            % (
                "." * level,
                description,
                counters["n"],
                counters["current"],
                counters["min"],
                counters["max"],
                counters["mean"],
            )
        )
