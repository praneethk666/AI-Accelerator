from concurrent.futures import ThreadPoolExecutor, TimeoutError

class TimeoutException(Exception):
    pass

def run_with_timeout(func, timeout_seconds, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except TimeoutError:
            raise TimeoutException(
                f"Function {func.__name__} timed out after {timeout_seconds}s"
            )