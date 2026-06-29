import contextvars
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class TimeoutException(Exception):
    pass

def run_with_timeout(func, timeout_seconds, *args, **kwargs):
    # Run in a copy of the caller's context so ContextVars (notably the token-usage
    # sink) propagate into the timeout thread — otherwise usage.record() inside
    # `func` sees no sink and the call's tokens are silently dropped from the totals.
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ctx.run, func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except TimeoutError:
            raise TimeoutException(
                f"Function {func.__name__} timed out after {timeout_seconds}s"
            )