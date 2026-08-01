import pandas as pd
import pytest

from biosim import ExperimentalDataError, load_experimental_csv


def test_load_experimental_csv_with_all_columns(tmp_path):
    path = tmp_path / "all.csv"
    pd.DataFrame(
        {
            "t": [1, 0],
            "X": [0.5, 0.1],
            "S": [15.0, 20.0],
            "P": [3.0, 0.0],
            "OTR": [5.0, 4.0],
        }
    ).to_csv(path, index=False)

    df = load_experimental_csv(path)

    assert list(df.columns) == ["t", "X", "S", "P", "OTR"]
    assert df["t"].tolist() == [0, 1]
    assert df["X"].tolist() == [0.1, 0.5]


def test_load_experimental_csv_only_subset_of_columns(tmp_path):
    path = tmp_path / "subset.csv"
    pd.DataFrame({"t": [0, 1], "P": [0.0, 3.0]}).to_csv(path, index=False)

    df = load_experimental_csv(path)

    assert list(df.columns) == ["t", "P"]


def test_load_experimental_csv_converts_od_to_x(tmp_path):
    path = tmp_path / "od.csv"
    pd.DataFrame({"t": [0, 1, 2], "OD": [0.1, 0.2, 0.4]}).to_csv(path, index=False)

    df = load_experimental_csv(path, od_conversion_factor=2.5)

    assert list(df.columns) == ["t", "X"]
    assert df["X"].tolist() == pytest.approx([0.25, 0.5, 1.0])


def test_load_experimental_csv_prefers_explicit_x_over_od(tmp_path):
    path = tmp_path / "both.csv"
    pd.DataFrame({"t": [0, 1], "X": [1.0, 2.0], "OD": [0.1, 0.2]}).to_csv(path, index=False)

    df = load_experimental_csv(path, od_conversion_factor=100.0)

    assert list(df.columns) == ["t", "X"]
    assert df["X"].tolist() == [1.0, 2.0]


def test_load_experimental_csv_missing_time_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"time": [0, 1], "X": [0.1, 0.2]}).to_csv(path, index=False)

    with pytest.raises(ExperimentalDataError, match="required `t` column"):
        load_experimental_csv(path)


def test_load_experimental_csv_no_recognized_measurement_columns_raises(tmp_path):
    path = tmp_path / "empty_measurements.csv"
    pd.DataFrame({"t": [0, 1], "notes": ["a", "b"]}).to_csv(path, index=False)

    with pytest.raises(ExperimentalDataError, match="no recognized measurement columns"):
        load_experimental_csv(path)
