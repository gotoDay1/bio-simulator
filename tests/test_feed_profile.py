import pandas as pd
import pytest

from biosim import FeedProfileError, InvalidParameterError, load_feed_profile_csv


def test_load_feed_profile_csv_builds_step_hold_callable(tmp_path):
    path = tmp_path / "feed.csv"
    pd.DataFrame(
        {"time": [5.0, 0.0, 2.0], "feed_rate": [0.03, 0.01, 0.05]}
    ).to_csv(path, index=False)

    fn = load_feed_profile_csv(path)

    assert fn(0.0) == pytest.approx(0.01)
    assert fn(1.0) == pytest.approx(0.01)
    assert fn(2.0) == pytest.approx(0.05)
    assert fn(4.999) == pytest.approx(0.05)
    assert fn(5.0) == pytest.approx(0.03)
    assert fn(100.0) == pytest.approx(0.03)


def test_load_feed_profile_csv_missing_columns_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"t": [0, 1], "rate": [0.1, 0.2]}).to_csv(path, index=False)

    with pytest.raises(FeedProfileError, match="missing required column"):
        load_feed_profile_csv(path)


def test_load_feed_profile_csv_empty_raises(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame({"time": [], "feed_rate": []}).to_csv(path, index=False)

    with pytest.raises(FeedProfileError, match="no valid"):
        load_feed_profile_csv(path)


def test_load_feed_profile_csv_duplicate_times_raises(tmp_path):
    path = tmp_path / "dup.csv"
    pd.DataFrame({"time": [0.0, 0.0], "feed_rate": [0.1, 0.2]}).to_csv(path, index=False)

    with pytest.raises(InvalidParameterError):
        load_feed_profile_csv(path)


def test_load_feed_profile_csv_negative_rate_raises(tmp_path):
    path = tmp_path / "neg.csv"
    pd.DataFrame({"time": [0.0, 1.0], "feed_rate": [0.1, -0.2]}).to_csv(path, index=False)

    with pytest.raises(InvalidParameterError):
        load_feed_profile_csv(path)
