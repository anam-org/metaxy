from functools import wraps

import pytest


def skip_exception(exception: type[Exception], reason: str):
    # Func below is the real decorator and will receive the test function as param
    def decorator_func(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Try to run the test
                return f(*args, **kwargs)
            except exception:
                # If exception of given type happens
                # just swallow it and raise pytest.Skip with given reason
                pytest.skip(f"skipped {exception.__name__}: {reason}")

        return wrapper

    return decorator_func
