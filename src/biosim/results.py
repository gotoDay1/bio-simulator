from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class SimulationResults:
    """Wraps the simulation output as a pandas DataFrame (columns: t, X, S, P, [C_O2], V, mu, OUR, [OTR])."""

    data: pd.DataFrame

    def to_csv(self, path: str) -> None:
        self.data.to_csv(path, index=False)

    def to_plotly_figure(self) -> go.Figure:
        plot_columns = [c for c in self.data.columns if c != "t"]
        fig = make_subplots(
            rows=len(plot_columns),
            cols=1,
            shared_xaxes=True,
            subplot_titles=plot_columns,
        )
        for i, col in enumerate(plot_columns, start=1):
            fig.add_trace(
                go.Scatter(x=self.data["t"], y=self.data[col], mode="lines", name=col),
                row=i,
                col=1,
            )
        fig.update_xaxes(title_text="time (h)", row=len(plot_columns), col=1)
        fig.update_layout(height=250 * len(plot_columns), showlegend=False)
        return fig
