from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Sequence
__all__ = ['type_mismatch_msg', 'type_mismatch_seq_msg',
           'value_mismatch_msg', 'value_not_in_list_msg',
           'obj_mismatch_msg']


def type_mismatch_msg(obj_name: str, obj: Any,
                      l_typ: type | str | list[type | str]):
    assert isinstance(obj_name, str)
    typ2str: dict[type, str] = {
        int: "an integer",
        float: "a float number",
        bool: "a boolean",
        str: "a string",
        list: "a list",
        tuple: "a tuple",
        dict: "a dict",
    }

    if isinstance(l_typ, (type, str)):
        l_typ = [l_typ, ]
    l_typ = list(l_typ)

    l_typ_str = ''
    for typ in l_typ:
        if typ in typ2str:
            typ = typ2str[typ]
        elif isinstance(typ, type):
            typ = f"a '{typ}' instance"
        if l_typ_str != '':
            typ = 'or ' + typ
        l_typ_str += typ
    return f"'{obj_name}' must be {l_typ_str}. " \
           f"got '{type(obj)}' instance instead."


def type_mismatch_seq_msg(seq_name: str, seq: Sequence[Any], typ: type | str):
    assert isinstance(seq_name, str)
    assert isinstance(typ, type)
    seq = list(seq)
    typ_seq = f"{list(type(obj) for obj in seq)}"[1:-1]
    if isinstance(typ, type):
        typ = f"'{typ}' instances"
    return f"'{seq_name}' must be a sequence of {typ}. " \
           f"got: {typ_seq}."


def value_mismatch_msg(obj_name: str, obj: Any, value: Any):
    assert isinstance(obj_name, str)
    return f"'{obj_name}' must be equal to {value}. " \
           f"got {obj_name} = {obj} (type '{type(obj)}') instead."


def value_not_in_list_msg(obj_name: str, obj: Any, l_values: Sequence[Any]):
    assert isinstance(obj_name, str)
    l_values = list(l_values)
    return f"'{obj_name}' must be one of the following: " \
           f"{str(l_values)[1:, -1]}. " \
           f"got {obj_name} = {obj}"


def obj_mismatch_msg(obj1_name: str, obj1: Any,
                     obj2_name: str, obj2: Any):
    assert isinstance(obj1_name, str)
    assert isinstance(obj2_name, str)
    return f"'{obj1_name}' and '{obj2_name}' must refer to the same object. " \
           f"got '{repr(obj1)}' and '{repr(obj2)}'."
