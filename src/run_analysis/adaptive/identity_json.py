"""Read-only dataclass JSON identities without an unnecessary deepcopy pass.

The JSON encoder recursively visits the same named fields and containers that
``dataclasses.asdict`` would copy. This is ONLY for immediate canonical identity
encoding: it is not a replacement for detached snapshots or mutable API payloads.
The caller retains its original separators, date encoder and strict NaN policy.
No floats are recomputed and no identity/version field is removed.
"""
from dataclasses import fields,is_dataclass
from functools import cached_property,lru_cache
from datetime import date,datetime,timezone
from zoneinfo import ZoneInfo
import json


@lru_cache(maxsize=128)
def _field_names(cls):
    # Dataclass schemas are fixed; cache type metadata, not athlete values.
    return tuple(field.name for field in fields(cls))


def identity_fields(value):
    if not is_dataclass(value) or isinstance(value,type):
        raise TypeError('identity_fields() requires a dataclass instance')
    return {name:getattr(value,name) for name in _field_names(type(value))}


def identity_dumps(value,**kwargs):
    original=kwargs.pop('default',None)
    def encode(item):
        if is_dataclass(item) and not isinstance(item,type):
            return identity_fields(item)
        if original is not None:
            return original(item)
        return json.JSONEncoder().default(item)
    return json.dumps(value,default=encode,**kwargs)


def _immutable(value):
    if type(value) in (str,int,float,bool,type(None),date):
        return True
    if type(value) is datetime:
        return value.tzinfo is None or type(value.tzinfo) in (timezone,ZoneInfo)
    if type(value) is tuple:
        return all(_immutable(item) for item in value)
    if is_dataclass(value) and not isinstance(value,type) and value.__dataclass_params__.frozen:
        return all(_immutable(getattr(value,name)) for name in _field_names(type(value)))
    return False


class FrozenIdentity:
    """Opt-in object-owned JSON reuse for recursively immutable physical rows.

    No equality-based cache: 1 vs 1.0, signed zero and datetime offsets retain
    their original bytes. Dict/list-bearing diagnostics use the ordinary
    encoder. Replaced/corrected records own a new cache; no athlete-wide memo.
    """
    @cached_property
    def _identity_fragments(self):
        return {} if _immutable(self) else None

    def _identity_fragment(self,*,default,compact):
        cache=self._identity_fragments
        key=(default,compact)
        if cache is not None and key in cache:
            return cache[key]
        result=identity_dumps(self,sort_keys=True,allow_nan=False,default=default,
            separators=(',',':') if compact else (', ',': '))
        if cache is not None and len(cache)<4:
            cache[key]=result
        return result


def identity_sequence_dumps(rows,*,default,compact=False):
    """Join complete standard-encoder fragments with identical array syntax."""
    separator=',' if compact else ', '
    return '['+separator.join(row._identity_fragment(default=default,compact=compact)
        if isinstance(row,FrozenIdentity) else identity_dumps(row,sort_keys=True,
            allow_nan=False,default=default,separators=(',',':') if compact else (', ',': '))
        for row in rows)+']'


def identity_object_dumps(value,*,fragments,default,compact=False):
    """Canonical string-keyed object; supplied fragments are encoder output.

    Internal identity construction only, not raw JSON supplied by an upload.
    Every nonfragment value still uses the original strict standard encoder.
    """
    if any(type(key) is not str for key in value) or not fragments.keys()<=value.keys():
        raise ValueError('Explicit string identity fields required')
    item_separator,key_separator=(',',':') if compact else (', ',': ')
    options=dict(sort_keys=True,allow_nan=False,default=default,
        separators=(item_separator,key_separator))
    return '{'+item_separator.join(json.dumps(key)+key_separator+(
        fragments[key] if key in fragments else identity_dumps(value[key],**options))
        for key in sorted(value))+'}'
