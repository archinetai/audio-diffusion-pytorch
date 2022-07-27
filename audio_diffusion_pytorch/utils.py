from inspect import isfunction
from typing import Callable, Optional, TypeVar, Union

from typing_extensions import TypeGuard

T = TypeVar("T")


def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None


def iff(condition: bool, value: T) -> Optional[T]:
    return value if condition else None


def is_sequence(obj: T) -> TypeGuard[Union[list, tuple]]:
    return isinstance(obj, list) or isinstance(obj, tuple)


def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d
