# Vendored from MASSACHUSETTS INSTITUTE OF TECHNOLOGY's responsible-ai-toolbox
# (rai_toolbox._utils), Copyright 2023 MIT. SPDX-License-Identifier: MIT
#
# Only the type check of `value_check` is needed by mushin; it is vendored here
# so that this package does not depend on the (unmaintained) rai_toolbox package.
from typing import Any, TypeVar, cast

T = TypeVar("T", bound=Any)


def _safe_name(x: Any) -> str:
    return getattr(x, "__name__", str(x))


def value_check(
    name: str,
    value: T,
    *,
    type_: type | tuple[type, ...],
    optional: bool = False,
) -> T:
    """
    For internal use only.

    Used to check the type of `value`.

    Examples
    --------
    >>> value_check("x", 1, type_=str)
    TypeError: `x` must be of type(s) `str`, got 1 (type: int)

    >>> value_check("x", None, type_=str, optional=True)  # ok

    Raises
    ------
    TypeError"""
    assert isinstance(name, str), name

    if optional and value is None:
        return value

    if not isinstance(value, type_):
        raise TypeError(
            f"`{name}` must be {'None or ' if optional else ''}of type(s) "
            f"`{_safe_name(type_)}`, got {value} (type: {_safe_name(type(value))})"
        )
    return cast(T, value)
