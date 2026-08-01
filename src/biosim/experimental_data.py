"""Loader for measured experimental data (a single CSV), for overlay comparison against
simulation results.

The CSV uses a fixed column schema: a required `t` (time, h) column plus any subset of the
optional measurement columns `X` (DCW, g/L), `OD` (turbidity, converted to `X`), `S`
(substrate, g/L), `P` (product, g/L), `OTR`. Columns that aren't present are simply absent
from the returned DataFrame - callers (SimulationResults.to_plotly_figure) skip overlaying
whatever measurement isn't there, so one file can carry any subset of X/OD, S, P, OTR.
"""

import pandas as pd

from biosim.exceptions import ExperimentalDataError

TIME_COLUMN = "t"
OPTIONAL_MEASUREMENT_COLUMNS = ["X", "OD", "S", "P", "OTR"]


def load_experimental_csv(path_or_buffer, od_conversion_factor: float = 1.0) -> pd.DataFrame:
    """Load a measured-data CSV for overlay comparison with simulation results.

    Fixed schema: a required `t` (h) column, plus any subset of the optional columns
    `X` (DCW, g/L), `OD` (turbidity; converted to `X = od_conversion_factor * OD` when
    `X` itself isn't present), `S` (substrate, g/L), `P` (product, g/L), `OTR`.

    Returns a DataFrame with `t` plus whichever of `X`/`S`/`P`/`OTR` were present in the
    source file - missing ones are simply absent, so the caller can overlay each measured
    column onto the matching simulated one and skip anything not provided.
    """
    df = pd.read_csv(path_or_buffer)
    if TIME_COLUMN not in df.columns:
        raise ExperimentalDataError(
            f"Experimental data CSV is missing the required `{TIME_COLUMN}` column. "
            f"Found columns {list(df.columns)}."
        )

    df = df.copy()
    if "OD" in df.columns and "X" not in df.columns:
        df["X"] = df["OD"] * od_conversion_factor

    measurement_columns = [c for c in ("X", "S", "P", "OTR") if c in df.columns]
    if not measurement_columns:
        raise ExperimentalDataError(
            "Experimental data CSV has no recognized measurement columns (expected at "
            f"least one of X, OD, S, P, OTR); found columns {list(df.columns)}."
        )

    result = df[[TIME_COLUMN, *measurement_columns]].dropna(subset=[TIME_COLUMN])
    return result.sort_values(TIME_COLUMN).reset_index(drop=True)
