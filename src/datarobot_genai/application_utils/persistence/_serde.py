# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-field (de)serialisation for ``DRSession`` metadata and ``DREvent`` body fields.

The Memory Service ORM stores declared fields as JSON — session ``metadata.<f>``
and event ``body.<f>``.  The base serializer only handled JSON primitives, so a
nested Pydantic model (e.g. ``list[ToolCall]``) would either raise on write or come
back as a raw ``dict`` on read.

These helpers build one cached :class:`pydantic.TypeAdapter` per ``(model class,
field name)`` and use it to:

* **write** — dump any Pydantic-expressible value (nested models, lists, optionals,
  enums, dicts) to a JSON-compatible Python object (scalars pass through unchanged);
* **read** — validate/coerce the raw wire value back into the declared type, falling
  back to the raw value when the wire payload cannot be validated (robustness first —
  the server is the source of truth and reads must never raise).
"""

from __future__ import annotations

from typing import Any
from weakref import WeakKeyDictionary

from pydantic import BaseModel
from pydantic import TypeAdapter
from pydantic import ValidationError

# One TypeAdapter per (model class, field name).  Keyed weakly on the class so the
# adapters are collected together with a transient subclass; ``TypeAdapter``
# construction is not free, so this cache keeps the hot (de)serialisation paths cheap.
_ADAPTER_CACHE: WeakKeyDictionary[type, dict[str, TypeAdapter[Any]]] = WeakKeyDictionary()


def field_type_adapter(model_cls: type[BaseModel], field_name: str) -> TypeAdapter[Any]:
    """Return a cached :class:`~pydantic.TypeAdapter` for one declared field.

    Parameters
    ----------
    model_cls : type[BaseModel]
        The ``DRSession``/``DREvent`` subclass owning the field.
    field_name : str
        Name of a declared field on *model_cls*.

    Returns
    -------
    TypeAdapter
        Adapter built from the field's annotation, built once and reused.
    """
    per_class = _ADAPTER_CACHE.get(model_cls)
    if per_class is None:
        per_class = {}
        _ADAPTER_CACHE[model_cls] = per_class

    adapter = per_class.get(field_name)
    if adapter is None:
        annotation = model_cls.model_fields[field_name].annotation
        # A field always carries an annotation in practice; fall back to ``Any`` so a
        # pathological untyped field still round-trips rather than raising here.
        adapter = TypeAdapter(annotation if annotation is not None else Any)
        per_class[field_name] = adapter
    return adapter


def serialize_field(model_cls: type[BaseModel], field_name: str, value: Any) -> Any:
    """Serialise *value* to a JSON-compatible Python object for the wire.

    Scalars are returned unchanged (``TypeAdapter(str).dump_python("x", mode="json")``
    is ``"x"``); nested models, lists, optionals, enums and dicts are converted to
    their JSON representation.

    Parameters
    ----------
    model_cls : type[BaseModel]
        The model subclass owning the field.
    field_name : str
        Name of the declared field.
    value : Any
        The value to serialise.

    Returns
    -------
    Any
        A JSON-serialisable Python object.
    """
    return field_type_adapter(model_cls, field_name).dump_python(value, mode="json")


def deserialize_field(model_cls: type[BaseModel], field_name: str, raw: Any) -> Any:
    """Validate/coerce a raw wire value into the field's declared type.

    On any validation failure the raw value is returned unchanged so reads remain
    robust to malformed or foreign-written wire data (the server is the source of
    truth and reads must never raise).

    Parameters
    ----------
    model_cls : type[BaseModel]
        The model subclass owning the field.
    field_name : str
        Name of the declared field.
    raw : Any
        The raw value taken from the wire ``metadata``/``body`` object.

    Returns
    -------
    Any
        The coerced value, or *raw* unchanged when it cannot be validated.
    """
    try:
        return field_type_adapter(model_cls, field_name).validate_python(raw)
    except (ValidationError, TypeError, ValueError):
        return raw
