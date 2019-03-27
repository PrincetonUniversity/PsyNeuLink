import abc
import enum
import json
import numpy
import types

from psyneulink.core.globals.sampleiterator import SampleIterator
from psyneulink.core.globals.utilities import parse_valid_identifier

__all__ = [
    'JSONDumpable', 'PNLJSONEncoder',
]


class JSONDumpable:
    @property
    @abc.abstractmethod
    def _dict_summary(self):
        pass

    @property
    def json_summary(self):
        return json.dumps(
            self._dict_summary,
            sort_keys=True,
            indent=4,
            separators=(',', ': '),
            cls=PNLJSONEncoder
        )


class PNLJSONEncoder(json.JSONEncoder):
    """
        A `JSONEncoder
        <https://docs.python.org/3/library/json.html#json.JSONEncoder>`_
        that parses `_dict_summary <Component._dict_summary>` output
        into a more JSON-friendly format.
    """
    def default(self, o):
        from psyneulink.core.components.component import Component, ComponentsMeta

        if isinstance(o, type):
            if o.__module__ == 'builtins':
                # just give standard type, like float or int
                return f'{o.__name__}'
            elif o is numpy.ndarray:
                return f'{o.__module__}.array'
            else:
                return f'{o.__module__}.{o.__name__}'
        elif isinstance(o, (enum.Enum, types.FunctionType)):
            return str(o)
        elif o is NotImplemented:
            return None
        elif isinstance(o, Component):
            return o.name
        elif isinstance(o, ComponentsMeta):
            return o.__name__
        elif isinstance(o, SampleIterator):
            return f'{o.__class__.__name__}({repr(o.specification)})'
        elif isinstance(o, numpy.ndarray):
            return list(o)
        elif isinstance(o, numpy.random.RandomState):
            return f'numpy.random.RandomState({o.seed})'
        else:
            try:
                # convert numpy number type to python type
                return o.item()
            except AttributeError:
                pass

        return super().default(o)
