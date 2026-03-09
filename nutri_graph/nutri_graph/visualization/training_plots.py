import re
from pathlib import Path
from typing import Dict, List, Optional

import plotly.graph_objects as go


# -----------------------------
# Paper-style Plotly settings (match Colab)
# -----------------------------
PLOT_WIDTH  = 720
PLOT_HEIGHT = 420

FONT_FAMILY = "Arial"
BASE_FONT   = 20
TITLE_FONT  = 22
AXIS_FONT   = 20
LEGEND_FONT = 18
LINE_WIDTH  = 4

GRID_COLOR  = "rgba(0,0,0,0.18)"
AXIS_COLOR  = "black"
BG_COLOR    = "white"


def _safe_slug(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def apply_paper_style(fig: go.Figure, title: str, xlab: str = "Epoch", ylab: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,

        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center",
            font=dict(family=FONT_FAMILY, size=TITLE_FONT, color="black")
        ),

        font=dict(family=FONT_FAMILY, size=BASE_FONT, color="black"),

        legend=dict(
            title=dict(text=""),
            font=dict(family=FONT_FAMILY, size=LEGEND_FONT, color="black"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=2
        ),

        margin=dict(l=70, r=20, t=70, b=60)
    )

    # Boxed axes + bold ticks + gridlines
    fig.update_xaxes(
        title=dict(text=f"<b>{xlab}</b>", font=dict(size=AXIS_FONT, color="black")),
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=3,
        linecolor=AXIS_COLOR,
        mirror=True,
        ticks="outside",
        tickwidth=3,
        tickcolor=AXIS_COLOR,
        tickfont=dict(size=AXIS_FONT, color="black")
    )
    fig.update_yaxes(
        title=dict(text=f"<b>{ylab}</b>", font=dict(size=AXIS_FONT, color="black")),
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=3,
        linecolor=AXIS_COLOR,
        mirror=True,
        ticks="outside",
        tickwidth=3,
        tickcolor=AXIS_COLOR,
        tickfont=dict(size=AXIS_FONT, color="black")
    )

    return fig


def save_fig(fig: go.Figure, base_name: str, out_dir: str, write_png: bool = True) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base = _safe_slug(base_name)
    html_path = out_path / f"{base}.html"
    fig.write_html(str(html_path))

    if write_png:
        png_path = out_path / f"{base}.png"
        try:
            fig.write_image(str(png_path), scale=3)
        except Exception as e:
            # Keep Colab-like behavior: HTML always saved; PNG best-effort
            print(f"[plot] PNG export skipped/failed for '{base_name}': {e}")


def plot_single_series(
    y: List[float],
    title: str,
    ylab: str,
    series_name: str,
    out_dir: str,
    write_png: bool = True
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=y,
        mode="lines",
        name=series_name,
        line=dict(width=LINE_WIDTH)
    ))
    apply_paper_style(fig, title, xlab="Epoch", ylab=ylab)
    save_fig(fig, title, out_dir=out_dir, write_png=write_png)


def plot_two_series(
    y1: List[float],
    name1: str,
    y2: List[float],
    name2: str,
    title: str,
    ylab: str,
    out_dir: str,
    write_png: bool = True
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y1, mode="lines", name=name1, line=dict(width=LINE_WIDTH)))
    fig.add_trace(go.Scatter(y=y2, mode="lines", name=name2, line=dict(width=LINE_WIDTH)))
    apply_paper_style(fig, title, xlab="Epoch", ylab=ylab)
    save_fig(fig, title, out_dir=out_dir, write_png=write_png)


def make_training_plots(history: Dict[str, List[float]], out_dir: str = "outputs/training", write_png: bool = True) -> None:
    """
    Matches Colab plot set:
      - Train Loss
      - Validation Regression (MAE + RMSE)
      - Validation Existence AUC
      - Learning Rate
    """

    plot_single_series(history["train_loss"], "Train Loss", "Loss", "Train Loss", out_dir, write_png=write_png)

    plot_two_series(
        history["val_mae"], "MAE",
        history["val_rmse"], "RMSE",
        "Validation Regression (log1p(amount))", "Metric",
        out_dir, write_png=write_png
    )

    plot_single_series(history["val_auc"], "Validation Existence AUC", "AUC", "Val AUC", out_dir, write_png=write_png)
    plot_single_series(history["lr"], "Learning Rate", "Learning Rate", "LR", out_dir, write_png=write_png)