# Vendored from MASSACHUSETTS INSTITUTE OF TECHNOLOGY's responsible-ai-toolbox
# (rai_toolbox._utils), Copyright 2023 MIT. SPDX-License-Identifier: MIT
#
# Only `value_check` and its helpers are needed by mushin; they are vendored here
# so that this package does not depend on the (unmaintained) rai_toolbox package.
from numbers import Real
from typing import Any, TypeVar, cast

T = TypeVar("T", bound=Any)


class Unsatisfiable(AssertionError):  # pragma: no cover
    pass


def _safe_name(x: Any) -> str:
    return getattr(x, "__name__", str(x))


def value_check(
    name: str,
    value: T,
    *,
    type_: type | tuple[type, ...] = Real,
    min_: int | float | None = None,
    max_: int | float | None = None,
    incl_min: bool = True,
    incl_max: bool = True,
    optional: bool = False,
    lower_name: str = "",
    upper_name: str = "",
) -> T:
    """
    For internal use only.

    Used to check the type of `value`. Numerical types can also be bound-checked.

    Examples
    --------
    >>> value_check("x", 1, type_=str)
    TypeError: `x` must be of type(s) `str`, got 1 (type: int)

    >>> value_check("x", 1, min_=20)
    ValueError: `x` must satisfy 20 <= x  Got: 1

    >>> value_check("x", 1, min_=1, incl_min=False)
    ValueError: `x` must satisfy 1 < x  Got: 1

    >>> value_check("x", 1, min_=1, incl_min=True) # ok
    1
    >>> value_check("x", 0.0, min_=-10, max_=10)  # ok
    0.0

    Raises
    ------
    TypeError, ValueError"""
    # check internal params
    assert isinstance(name, str), name
    assert min_ is None or isinstance(min_, (int, float)), min_
    assert max_ is None or isinstance(max_, (int, float)), max_
    assert isinstance(incl_min, bool), incl_min
    assert isinstance(incl_max, bool), incl_max

    if optional and value is None:
        return value

    if not isinstance(value, type_):
        raise TypeError(
            f"`{name}` must be {'None or' if optional else ''}of type(s) "
            f"`{_safe_name(type_)}`, got {value} (type: {_safe_name(type(value))})"
        )

    if min_ is not None and max_ is not None:
        if incl_max and incl_min:
            if not (min_ <= max_):
                raise Unsatisfiable(f"{min_} <= {max_}")
        elif not min_ < max_:
            raise Unsatisfiable(f"{min_} < {max_}")

    min_satisfied = (
        (min_ <= value if incl_min else min_ < value) if min_ is not None else True
    )
    max_satisfied = (
        (value <= max_ if incl_max else value < max_) if max_ is not None else True
    )

    if not min_satisfied or not max_satisfied:
        lsymb = "<=" if incl_min else "<"
        rsymb = "<=" if incl_max else "<"

        err_msg = f"`{name}` must satisfy"

        if min_ is not None:
            if lower_name:  # pragma: no cover
                min_ = f"{lower_name}(= {min_})"  # type: ignore
            err_msg += f" {min_} {lsymb}"

        err_msg += f" {name}"

        if max_ is not None:
            if upper_name:
                max_ = f"{upper_name}(= {max_})"  # type: ignore
            err_msg += f" {rsymb} {max_}"

        err_msg += f"  Got: {value}"

        raise ValueError(err_msg)
    return cast(T, value)
