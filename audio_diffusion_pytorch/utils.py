from inspect import isfunction


def exists(val) -> bool:
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
