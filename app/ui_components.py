import dataclasses
import types
import typing

import streamlit as st

from biosim import (
    FeedProfileError,
    InvalidParameterError,
    constant_feed,
    exponential_feed,
    load_feed_profile_csv,
    step_feed,
)
from biosim.fitting import ParameterSpec

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


def render_fit_param_inputs(model_cls: type, key_prefix: str) -> list[ParameterSpec]:
    """Render fixed/free controls for each dataclass field and return the resulting specs.

    Optional[float] fields (e.g. Yps) get a 3-way choice (unused/fixed/free) instead of the
    regular 2-way (fixed/free), since "unused" (None) is a meaningful state for them.
    """
    specs: list[ParameterSpec] = []
    for f in dataclasses.fields(model_cls):
        if f.name in SKIP_FIELDS:
            continue

        has_default = f.default is not dataclasses.MISSING
        fallback = 0.1
        default_val = float(f.default) if has_default and f.default is not None else fallback
        is_optional = _is_optional_float(f.type)

        mode_options = ["固定", "自由(推定)"]
        if is_optional:
            mode_options = ["未使用 (None)", *mode_options]
        mode = st.radio(f.name, mode_options, key=f"{key_prefix}_{f.name}_mode")

        if mode == "未使用 (None)":
            specs.append(ParameterSpec(name=f.name, fixed=True, value=None, is_optional=True))
        elif mode == "固定":
            value = st.number_input(
                f"{f.name} (固定値)", value=default_val, key=f"{key_prefix}_{f.name}_fixed"
            )
            specs.append(
                ParameterSpec(name=f.name, fixed=True, value=value, is_optional=is_optional)
            )
        else:
            guess = st.number_input(
                f"{f.name} (初期推定値)", value=default_val, key=f"{key_prefix}_{f.name}_guess"
            )
            lower = st.number_input(
                f"{f.name} (下限)", value=0.0, key=f"{key_prefix}_{f.name}_lower"
            )
            unbounded = st.checkbox(
                f"{f.name} 上限なし", value=True, key=f"{key_prefix}_{f.name}_unbounded"
            )
            if unbounded:
                upper = float("inf")
            else:
                upper = st.number_input(
                    f"{f.name} (上限)",
                    value=max(default_val * 10, 1.0),
                    key=f"{key_prefix}_{f.name}_upper",
                )
            specs.append(
                ParameterSpec(
                    name=f.name,
                    fixed=False,
                    value=guess,
                    lower_bound=lower,
                    upper_bound=upper,
                    is_optional=is_optional,
                )
            )
    return specs


def render_feed_profile_inputs(key_prefix: str) -> dict:
    """Render the fed-batch feed-profile picker (rate profile + S_feed), shared by both
    the simulation page and the fitting page.

    Returns {"feed_rate_fn": Callable[[float], float] | None, "S_feed": float}.
    feed_rate_fn is None only when the "csv" profile is selected and no valid file has
    been uploaded yet (no file chosen, or the uploaded CSV failed to parse - in which
    case an st.error is already shown here). Callers must treat None as "not ready to
    run" and avoid constructing FedBatch(feed_rate_fn=None, ...).
    """
    st.subheader("流加培養パラメータ")
    profile_choice = st.selectbox(
        "フィードプロファイル",
        ["constant", "step", "exponential", "csv"],
        key=f"{key_prefix}_profile",
    )

    feed_rate_fn = None
    if profile_choice == "constant":
        rate = st.number_input(
            "フィード流量 F (L/h)", value=0.05, min_value=0.0, key=f"{key_prefix}_rate_const"
        )
        feed_rate_fn = constant_feed(rate)
    elif profile_choice == "step":
        t_start = st.number_input(
            "開始時刻 t_start (h)", value=2.0, min_value=0.0, key=f"{key_prefix}_t_start"
        )
        rate = st.number_input(
            "フィード流量 F (L/h)", value=0.05, min_value=0.0, key=f"{key_prefix}_rate_step"
        )
        feed_rate_fn = step_feed(t_start=t_start, rate=rate)
    elif profile_choice == "exponential":
        F0 = st.number_input(
            "初期フィード流量 F0 (L/h)", value=0.02, min_value=0.0, key=f"{key_prefix}_F0"
        )
        mu_set = st.number_input(
            "目標比増殖速度 mu_set (1/h)", value=0.1, key=f"{key_prefix}_mu_set"
        )
        feed_rate_fn = exponential_feed(F0=F0, mu_set=mu_set)
    else:  # "csv"
        feed_file = st.file_uploader(
            "フィードプロファイルCSV（列名: time, feed_rate。段階的な流加を複数ブレークポイントで指定）",
            type="csv",
            key=f"{key_prefix}_csv",
        )
        if feed_file is not None:
            try:
                feed_rate_fn = load_feed_profile_csv(feed_file)
            except (FeedProfileError, InvalidParameterError) as e:
                st.error(str(e))
        else:
            st.caption("time, feed_rate の2列を持つCSVをアップロードしてください。")

    S_feed = st.number_input(
        "フィード基質濃度 S_feed (g/L)", value=100.0, min_value=0.0, key=f"{key_prefix}_S_feed"
    )
    return {"feed_rate_fn": feed_rate_fn, "S_feed": S_feed}
