from sys import version_info

if version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

from beartype.typing import (
    Union as Union,
    Type as Type,
    Callable as Callable,
    Optional as Optional,
    Any as Any,
    Tuple as Tuple,
    List as List,
    Dict as Dict,
    Iterable as Iterable,
    Set as Set,
)
