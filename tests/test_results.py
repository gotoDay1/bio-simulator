import pandas as pd

from biosim import SimulationResults


def test_run_produces_expected_columns(standard_batch_simulation):
    results = standard_batch_simulation.run()
    for col in ("t", "X", "S", "P", "V", "mu", "OUR"):
        assert col in results.data.columns
    assert "C_O2" not in results.data.columns
    assert "OTR" not in results.data.columns
    assert len(results.data) == standard_batch_simulation.n_points


def test_to_csv_writes_readable_file(tmp_path, standard_batch_simulation):
    results = standard_batch_simulation.run()
    path = tmp_path / "out.csv"
    results.to_csv(str(path))
    loaded = pd.read_csv(path)
    assert list(loaded.columns) == list(results.data.columns)
    assert len(loaded) == len(results.data)


def test_to_plotly_figure_has_one_subplot_per_column(standard_batch_simulation):
    results = standard_batch_simulation.run()
    fig = results.to_plotly_figure()
    n_columns = len([c for c in results.data.columns if c != "t"])
    assert len(fig.data) == n_columns


def test_simulation_results_wraps_arbitrary_dataframe():
    df = pd.DataFrame({"t": [0, 1], "X": [0.1, 0.2]})
    results = SimulationResults(df)
    assert results.data is df
