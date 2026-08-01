import dataclasses
import types
import typing

import streamlit as st

SKIP_FIELDS = {"feed_rate_fn"}


def _is_optional_float(field_type) -> bool:
    if isinstance(field_type, types.UnionType):
        args = typing.get_args(field_type)
        return float in args and type(None) in args
    return False


def render_param_inputs(model_cls: type, key_prefix: str) -> dict:
    """Render one st.number_input per dataclass field and return kwargs to construct model_cls.

    Fields in SKIP_FIELDS are omitted (the caller supplies them separately, e.g.
    a feed-rate callable built from a dedicated profile picker). Optional[float]
    fields (e.g. Yps) get a checkbox to opt in; unchecked means None (disabled).
    """
    kwargs: dict = {}
    for f in dataclasses.fields(model_cls):
        if f.name in SKIP_FIELDS:
            continue

        has_default = f.default is not dataclasses.MISSING
        fallback = 0.1
        default_val = float(f.default) if has_default and f.default is not None else fallback

        if _is_optional_float(f.type):
            enabled = st.checkbox(
                f"{f.name} を使用する", value=False, key=f"{key_prefix}_{f.name}_enabled"
            )
            if enabled:
                kwargs[f.name] = st.number_input(
                    f.name, value=default_val, key=f"{key_prefix}_{f.name}"
                )
            else:
                kwargs[f.name] = None
        else:
            kwargs[f.name] = st.number_input(
                f.name, value=default_val, key=f"{key_prefix}_{f.name}"
            )
    return kwargs
