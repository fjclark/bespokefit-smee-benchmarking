"""Analyse RBFE benchmark results using cinnabar, with interactive plots and statistical tests."""

from pathlib import Path

import bokeh.io
import bokeh.models
import bokeh.plotting
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotmol
from adjustText import adjust_text
from openff.units import unit
from rdkit import Chem
from scipy.stats import binomtest, wilcoxon

matplotlib.use("Agg")

from cinnabar.compare import compare_and_rank_results
from cinnabar.femap import FEMap
from cinnabar.stats import bootstrap_statistic

TARGETS = ["cdk2", "jnk1", "tyk2"]
FORCE_FIELDS = ["presto", "default"]
FF_DISPLAY = {"presto": "Presto", "default": "OpenFF 1.3.1"}
SDF_MAP = {"cdk2": "CDK2", "jnk1": "JNK1", "tyk2": "TYK2"}
NBOOTSTRAP = 10_000

EXCLUDE_LIGANDS = {"tyk2": ["ejm_44"]}

# SMARTS for cyclopropyl directly bonded to carbonyl carbon
CYCLOPROPYL_CARBONYL_SMARTS = "[C;R1]1[C;R1][C;R1]1C(=O)"

# Column order for panel plots
PANEL_TARGET_ORDER = ["cdk2", "tyk2", "jnk1"]
# Row order: default (OpenFF) on top, presto on bottom
PANEL_FF_ORDER = ["default", "presto"]

DDG_STATISTICS = ["RMSE", "MUE"]
DG_STATISTICS = ["RMSE", "MUE", "R2", "KTAU"]

FIGSIZE = 3.96


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_smiles(target: str, ligands_dir: Path) -> dict[str, str]:
    sdf_path = ligands_dir / f"{SDF_MAP[target]}_ligands.sdf"
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mapping = {}
    for mol in suppl:
        if mol is not None:
            name = mol.GetProp("_Name").strip()
            mapping[name] = Chem.MolToSmiles(mol)
    return mapping


def load_predictions(target: str, ff: str, raw_data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(raw_data_dir / f"predictions_{target}_{ff}.csv")


def load_experimental(target: str, raw_data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_data_dir / f"{target}_exp.csv")
    df.columns = [c.strip() for c in df.columns]
    if "Ligand name" in df.columns:
        df = df.rename(columns={"Ligand name": "Ligand", "Exp. dG (kcal/mol)": "expt_DG"})
    df["Ligand"] = df["Ligand"].astype(str).str.strip()
    return df


def filter_predictions(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    mask = ~df["Ligand1"].isin(exclude) & ~df["Ligand2"].isin(exclude)
    return df[mask].copy()


def filter_experimental(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    return df[~df["Ligand"].isin(exclude)].copy()


# ---------------------------------------------------------------------------
# FEMap construction
# ---------------------------------------------------------------------------


def build_femap(
    predictions_df: pd.DataFrame, exp_df: pd.DataFrame, source: str, run_mle: bool = True,
) -> FEMap:
    femap = FEMap()
    for _, row in predictions_df.iterrows():
        femap.add_relative_calculation(
            labelA=row["Ligand1"],
            labelB=row["Ligand2"],
            value=row["mean_calc_DDG"] * unit.kilocalorie_per_mole,
            uncertainty=row["sem_calc_DDG"] * unit.kilocalorie_per_mole,
            source=source,
        )
    for _, row in exp_df.iterrows():
        femap.add_experimental_measurement(
            label=row["Ligand"],
            value=row["expt_DG"] * unit.kilocalorie_per_mole,
            uncertainty=0.0 * unit.kilocalorie_per_mole,
            source="Experimental",
        )
    if run_mle:
        femap.generate_absolute_values()
    return femap


def build_combined_femap(
    predictions_by_ff: dict[str, pd.DataFrame], exp_df: pd.DataFrame
) -> FEMap:
    femap = FEMap()
    for ff, pred_df in predictions_by_ff.items():
        for _, row in pred_df.iterrows():
            femap.add_relative_calculation(
                labelA=row["Ligand1"],
                labelB=row["Ligand2"],
                value=row["mean_calc_DDG"] * unit.kilocalorie_per_mole,
                uncertainty=row["sem_calc_DDG"] * unit.kilocalorie_per_mole,
                source=ff,
            )
    for _, row in exp_df.iterrows():
        femap.add_experimental_measurement(
            label=row["Ligand"],
            value=row["expt_DG"] * unit.kilocalorie_per_mole,
            uncertainty=0.0 * unit.kilocalorie_per_mole,
            source="Experimental",
        )
    return femap


# ---------------------------------------------------------------------------
# Helpers for extracting data from legacy graphs
# ---------------------------------------------------------------------------


def extract_edge_data(graph) -> dict[str, np.ndarray]:
    exp_ddg, calc_ddg, calc_dddg, labels_a, labels_b = [], [], [], [], []
    for u, v, d in graph.edges(data=True):
        exp_ddg.append(d["exp_DDG"])
        calc_ddg.append(d["calc_DDG"])
        calc_dddg.append(d["calc_dDDG"])
        labels_a.append(u)
        labels_b.append(v)
    return {
        "exp_ddg": np.array(exp_ddg),
        "calc_ddg": np.array(calc_ddg),
        "calc_dddg": np.array(calc_dddg),
        "labels_a": labels_a,
        "labels_b": labels_b,
    }


def extract_node_data(graph) -> dict[str, np.ndarray]:
    exp_dg, calc_dg, calc_ddg_node, labels = [], [], [], []
    for n, d in graph.nodes(data=True):
        if "calc_DG" not in d or d["calc_DG"] is None:
            continue
        exp_dg.append(d["exp_DG"])
        calc_dg.append(d["calc_DG"])
        calc_ddg_node.append(d["calc_dDG"])
        labels.append(n)
    exp_dg = np.array(exp_dg)
    calc_dg = np.array(calc_dg)
    calc_dg = calc_dg - calc_dg.mean() + exp_dg.mean()
    return {
        "exp_dg": exp_dg,
        "calc_dg": calc_dg,
        "calc_ddg": np.array(calc_ddg_node),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Section 1: Custom matplotlib correlation plots (cinnabar style)
# ---------------------------------------------------------------------------


def plot_correlation(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray | None,
    yerr: np.ndarray | None,
    ff_name: str,
    target_name: str,
    quantity: str,
    statistics: list[str],
    bootstrap_x_uncertainty: bool = False,
    bootstrap_y_uncertainty: bool = False,
    axis_padding: float = 0.5,
    xy_lim: tuple[float, float] | None = None,
) -> None:
    nsamples = len(x)
    if xy_lim is not None:
        ax_min, ax_max = xy_lim
    else:
        ax_min = min(x.min(), y.min()) - axis_padding
        ax_max = max(x.max(), y.max()) + axis_padding
    scale = [ax_min, ax_max]

    ax.set_xlim(scale)
    ax.set_ylim(scale)
    ax.set_aspect("equal")

    # x=y line
    ax.plot(scale, scale, "k:", zorder=0)

    # Shaded guidelines at +/- 0.5 and +/- 1.0 kcal/mol
    for delta in [1.0, 0.5]:
        ax.fill_between(
            scale,
            [ax_min - delta, ax_max - delta],
            [ax_min + delta, ax_max + delta],
            color="grey",
            alpha=0.2,
            zorder=0,
        )

    # Origin lines
    ax.plot([0, 0], scale, color="gray", linewidth=0.5, zorder=0)
    ax.plot(scale, [0, 0], color="gray", linewidth=0.5, zorder=0)

    # Error bars
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, color="gray", linewidth=0, elinewidth=2, zorder=1)

    # Scatter with coolwarm colormap based on |error|
    cm = plt.get_cmap("coolwarm")
    colors = cm(np.abs(x - y) / 2.372)
    ax.scatter(x, y, color=colors, s=20, marker="o", zorder=2, edgecolors="dimgrey", linewidths=0.7)

    ax.set_xlabel(f"Experimental {quantity}" + r" / $\mathrm{kcal\,mol^{-1}}$")
    ax.set_ylabel(f"Calculated {quantity}" + r" / $\mathrm{kcal\,mol^{-1}}$")

    # Bootstrap statistics
    stats_string = ""
    for statistic in statistics:
        s = bootstrap_statistic(
            x, y, xerr, yerr,
            statistic=statistic,
            include_true_uncertainty=bootstrap_x_uncertainty,
            include_pred_uncertainty=bootstrap_y_uncertainty,
        )
        stats_string += f"{statistic}:   {s['mle']:.2f} [95%: {s['low']:.2f}, {s['high']:.2f}] \n"

    title = f"{ff_name} \n {target_name} (N = {nsamples}) \n {stats_string}"
    ax.set_title(title, fontsize=10, loc="right", ha="right", family="monospace")


def run_cinnabar_plots(
    femaps: dict[str, dict[str, FEMap]],
    results_dir: Path,
    targets: list[str] | None = None,
    suffix: str = "",
) -> None:
    if targets is None:
        targets = TARGETS

    for target in targets:
        for ff in FORCE_FIELDS:
            graph = femaps[target][ff].to_legacy_graph()
            edge_data = extract_edge_data(graph)
            node_data = extract_node_data(graph)

            # DDG plot
            fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE))
            plot_correlation(
                ax, edge_data["exp_ddg"], edge_data["calc_ddg"],
                xerr=np.zeros_like(edge_data["exp_ddg"]),
                yerr=edge_data["calc_dddg"],
                ff_name=FF_DISPLAY[ff],
                target_name=target.upper(),
                quantity="ΔΔG",
                statistics=DDG_STATISTICS,
                bootstrap_y_uncertainty=True,
            )
            fig.tight_layout()
            fig.savefig(results_dir / f"{target}_{ff}_ddg{suffix}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            # DG plot
            fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE))
            plot_correlation(
                ax, node_data["exp_dg"], node_data["calc_dg"],
                xerr=np.zeros_like(node_data["exp_dg"]),
                yerr=node_data["calc_ddg"],
                ff_name=FF_DISPLAY[ff],
                target_name=target.upper(),
                quantity="ΔG",
                statistics=DG_STATISTICS,
                bootstrap_y_uncertainty=True,
            )
            fig.tight_layout()
            fig.savefig(results_dir / f"{target}_{ff}_dg{suffix}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"  Saved plots for {target}/{ff}{suffix}")


# ---------------------------------------------------------------------------
# Section 1b: Panel plots
# ---------------------------------------------------------------------------


def run_panel_plots(
    femaps: dict[str, dict[str, FEMap]],
    results_dir: Path,
    femaps_no_excl: dict[str, dict[str, FEMap]] | None = None,
) -> None:
    for quantity_label, extract_fn, stats, data_keys in [
        ("ddg", extract_edge_data, DDG_STATISTICS, ("exp_ddg", "calc_ddg", "calc_dddg")),
        ("dg", extract_node_data, DG_STATISTICS, ("exp_dg", "calc_dg", "calc_ddg")),
    ]:
        for variant_label, femap_set in [("", femaps), ("_no_ejm44", femaps_no_excl)]:
            if femap_set is None:
                continue

            fig, axes = plt.subplots(
                len(PANEL_FF_ORDER), len(PANEL_TARGET_ORDER),
                figsize=(FIGSIZE * len(PANEL_TARGET_ORDER), FIGSIZE * len(PANEL_FF_ORDER)),
                squeeze=False,
            )

            for row, ff in enumerate(PANEL_FF_ORDER):
                for col, target in enumerate(PANEL_TARGET_ORDER):
                    ax = axes[row, col]
                    graph = femap_set[target][ff].to_legacy_graph()
                    data = extract_fn(graph)
                    x_key, y_key, yerr_key = data_keys

                    plot_correlation(
                        ax, data[x_key], data[y_key],
                        xerr=np.zeros_like(data[x_key]),
                        yerr=data[yerr_key],
                        ff_name=FF_DISPLAY[ff],
                        target_name=target.upper(),
                        quantity="ΔΔG" if quantity_label == "ddg" else "ΔG",
                        statistics=stats,
                        bootstrap_y_uncertainty=True,
                    )

            fig.subplots_adjust(wspace=0.4, hspace=0.7)
            out = results_dir / f"panel_{quantity_label}{variant_label}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Section 2: Bootstrap statistics (n=10000)
# ---------------------------------------------------------------------------


def run_bootstrap_statistics(
    femaps: dict[str, dict[str, FEMap]],
    results_dir: Path,
    targets: list[str] | None = None,
    suffix: str = "",
) -> pd.DataFrame:
    if targets is None:
        targets = TARGETS

    rows = []
    ddg_metrics = ["MUE", "RMSE"]
    dg_metrics = ["MUE", "RMSE", "KTAU", "R2"]

    for target in targets:
        for ff in FORCE_FIELDS:
            graph = femaps[target][ff].to_legacy_graph()
            edge_data = extract_edge_data(graph)
            node_data = extract_node_data(graph)

            for metric in ddg_metrics:
                s = bootstrap_statistic(
                    edge_data["exp_ddg"],
                    edge_data["calc_ddg"],
                    dy_pred=edge_data["calc_dddg"],
                    statistic=metric,
                    nbootstrap=NBOOTSTRAP,
                    include_pred_uncertainty=True,
                )
                rows.append({
                    "target": target, "ff": ff, "level": "DDG", "metric": metric,
                    "mle": s["mle"], "mean": s["mean"], "ci_low": s["low"], "ci_high": s["high"],
                })

            for metric in dg_metrics:
                s = bootstrap_statistic(
                    node_data["exp_dg"],
                    node_data["calc_dg"],
                    dy_pred=node_data["calc_ddg"],
                    statistic=metric,
                    nbootstrap=NBOOTSTRAP,
                    include_pred_uncertainty=True,
                )
                rows.append({
                    "target": target, "ff": ff, "level": "DG", "metric": metric,
                    "mle": s["mle"], "mean": s["mean"], "ci_low": s["low"], "ci_high": s["high"],
                })

    df = pd.DataFrame(rows)
    fname = f"bootstrap_statistics{suffix}.csv"
    df.to_csv(results_dir / fname, index=False)
    print(df.to_string(index=False, float_format="%.3f"))
    return df


# ---------------------------------------------------------------------------
# Section 3: Interactive plots (plotmol + bokeh)
# ---------------------------------------------------------------------------


def _add_shaded_bands(figure, lo: float, hi: float) -> None:
    for delta, alpha in [(1.0, 0.1), (0.5, 0.15)]:
        figure.patch(
            x=[lo, hi, hi, lo],
            y=[lo + delta, hi + delta, hi - delta, lo - delta],
            fill_color="grey",
            fill_alpha=alpha,
            line_alpha=0,
        )


def _make_ddg_figure(
    edge_data: dict,
    smiles_map: dict[str, str],
    title: str,
    color: str = "steelblue",
    legend_label: str | None = None,
    figure: bokeh.plotting.figure | None = None,
) -> bokeh.plotting.figure:
    exp = edge_data["exp_ddg"]
    calc = edge_data["calc_ddg"]
    err = edge_data["calc_dddg"]
    combined_smiles = [
        f"{smiles_map.get(a, '')}.{smiles_map.get(b, '')}"
        for a, b in zip(edge_data["labels_a"], edge_data["labels_b"])
    ]
    labels = [
        f"{a} → {b}"
        for a, b in zip(edge_data["labels_a"], edge_data["labels_b"])
    ]

    if figure is None:
        all_vals = np.concatenate([exp, calc])
        pad = 0.5
        lo, hi = float(all_vals.min() - pad), float(all_vals.max() + pad)
        figure = bokeh.plotting.figure(
            tooltips=plotmol.default_tooltip_template(),
            x_axis_label="Experimental ΔΔG (kcal/mol)",
            y_axis_label="Calculated ΔΔG (kcal/mol)",
            title=title,
            width=700,
            height=700,
            x_range=(lo, hi),
            y_range=(lo, hi),
        )
        _add_shaded_bands(figure, lo, hi)
        figure.line(
            [lo, hi], [lo, hi], line_color="black", line_width=2, alpha=0.4, legend_label="x = y"
        )

    plotmol.scatter(
        figure,
        x=list(exp),
        y=list(calc),
        smiles=combined_smiles,
        marker_size=12,
        marker_color=color,
        legend_label=legend_label,
        custom_column_data={"label": labels, "err": list(err)},
    )

    upper = list(calc + err)
    lower = list(calc - err)
    source = bokeh.models.ColumnDataSource(
        data={"x": list(exp), "upper": upper, "lower": lower}
    )
    whisker = bokeh.models.Whisker(
        source=source, base="x", upper="upper", lower="lower",
        dimension="height", line_color=color, line_alpha=0.5,
    )
    figure.add_layout(whisker)

    return figure


def _make_dg_figure(
    node_data: dict,
    smiles_map: dict[str, str],
    title: str,
    color: str = "steelblue",
    legend_label: str | None = None,
    figure: bokeh.plotting.figure | None = None,
) -> bokeh.plotting.figure:
    exp = node_data["exp_dg"]
    calc = node_data["calc_dg"]
    err = node_data["calc_ddg"]
    smiles_list = [smiles_map.get(name, "") for name in node_data["labels"]]

    if figure is None:
        all_vals = np.concatenate([exp, calc])
        pad = 0.5
        lo, hi = float(all_vals.min() - pad), float(all_vals.max() + pad)
        figure = bokeh.plotting.figure(
            tooltips=plotmol.default_tooltip_template(),
            x_axis_label="Experimental ΔG (kcal/mol)",
            y_axis_label="Calculated ΔG (kcal/mol)",
            title=title,
            width=700,
            height=700,
            x_range=(lo, hi),
            y_range=(lo, hi),
        )
        _add_shaded_bands(figure, lo, hi)
        figure.line(
            [lo, hi], [lo, hi], line_color="black", line_width=2, alpha=0.4, legend_label="x = y"
        )

    plotmol.scatter(
        figure,
        x=list(exp),
        y=list(calc),
        smiles=smiles_list,
        marker_size=12,
        marker_color=color,
        legend_label=legend_label,
        custom_column_data={"label": list(node_data["labels"]), "err": list(err)},
    )

    upper = list(calc + err)
    lower = list(calc - err)
    source = bokeh.models.ColumnDataSource(
        data={"x": list(exp), "upper": upper, "lower": lower}
    )
    whisker = bokeh.models.Whisker(
        source=source, base="x", upper="upper", lower="lower",
        dimension="height", line_color=color, line_alpha=0.5,
    )
    figure.add_layout(whisker)

    return figure


def run_interactive_plots(
    femaps: dict[str, dict[str, FEMap]],
    smiles_maps: dict[str, dict[str, str]],
    results_dir: Path,
    targets: list[str] | None = None,
    suffix: str = "",
) -> None:
    if targets is None:
        targets = TARGETS
    ff_colors = {"presto": "steelblue", "default": "darkorange"}

    for target in targets:
        smiles_map = smiles_maps[target]

        # Per-FF plots
        for ff in FORCE_FIELDS:
            graph = femaps[target][ff].to_legacy_graph()
            edge_data = extract_edge_data(graph)
            node_data = extract_node_data(graph)
            title_suffix = f"{target.upper()} — {FF_DISPLAY[ff]}"

            fig_ddg = _make_ddg_figure(edge_data, smiles_map, f"ΔΔG {title_suffix}", color=ff_colors[ff])
            fig_ddg.legend.location = "top_left"
            fig_ddg.legend.click_policy = "hide"
            out = results_dir / f"{target}_{ff}_ddg_interactive{suffix}.html"
            bokeh.io.save(fig_ddg, filename=str(out), title=f"DDG {title_suffix}")
            print(f"  Saved {out.name}")

            fig_dg = _make_dg_figure(node_data, smiles_map, f"ΔG {title_suffix}", color=ff_colors[ff])
            fig_dg.legend.location = "top_left"
            fig_dg.legend.click_policy = "hide"
            out = results_dir / f"{target}_{ff}_dg_interactive{suffix}.html"
            bokeh.io.save(fig_dg, filename=str(out), title=f"DG {title_suffix}")
            print(f"  Saved {out.name}")

        # Overlay plots (both FFs on same figure)
        fig_ddg_overlay = None
        fig_dg_overlay = None
        for ff in FORCE_FIELDS:
            graph = femaps[target][ff].to_legacy_graph()
            edge_data = extract_edge_data(graph)
            node_data = extract_node_data(graph)

            fig_ddg_overlay = _make_ddg_figure(
                edge_data, smiles_map,
                f"ΔΔG {target.upper()} — overlay",
                color=ff_colors[ff],
                legend_label=FF_DISPLAY[ff],
                figure=fig_ddg_overlay,
            )
            fig_dg_overlay = _make_dg_figure(
                node_data, smiles_map,
                f"ΔG {target.upper()} — overlay",
                color=ff_colors[ff],
                legend_label=FF_DISPLAY[ff],
                figure=fig_dg_overlay,
            )

        fig_ddg_overlay.legend.location = "top_left"
        fig_ddg_overlay.legend.click_policy = "hide"
        out = results_dir / f"{target}_overlay_ddg_interactive{suffix}.html"
        bokeh.io.save(fig_ddg_overlay, filename=str(out), title=f"DDG overlay {target.upper()}")
        print(f"  Saved {out.name}")

        fig_dg_overlay.legend.location = "top_left"
        fig_dg_overlay.legend.click_policy = "hide"
        out = results_dir / f"{target}_overlay_dg_interactive{suffix}.html"
        bokeh.io.save(fig_dg_overlay, filename=str(out), title=f"DG overlay {target.upper()}")
        print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Section 4: Paired bootstrap comparison (per target)
# ---------------------------------------------------------------------------


def _reorient_comparison(comparison_df: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    """Reorient comparison so that the difference is model_a minus model_b."""
    df = comparison_df.copy()
    diff_cols = [c for c in df.columns if c.startswith("Diff in ")]
    for _, row in df.iterrows():
        m1, m2 = row["Model 1"], row["Model 2"]
        if m1 == model_b and m2 == model_a:
            idx = row.name
            df.at[idx, "Model 1"] = model_a
            df.at[idx, "Model 2"] = model_b
            for col in diff_cols:
                df.at[idx, col] = -df.at[idx, col]
            old_lo = df.at[idx, "CI Lower"]
            old_hi = df.at[idx, "CI Upper"]
            df.at[idx, "CI Lower"] = -old_hi
            df.at[idx, "CI Upper"] = -old_lo
    return df


def run_paired_comparison(
    combined_femaps: dict[str, FEMap],
    results_dir: Path,
    targets: list[str] | None = None,
    suffix: str = "",
) -> None:
    if targets is None:
        targets = TARGETS

    all_summary_ddg, all_comparison_ddg = [], []
    all_summary_dg, all_comparison_dg = [], []

    for target in targets:
        femap = combined_femaps[target]
        print(f"\n  --- {target.upper()} ---")

        summary_ddg, comparison_ddg = compare_and_rank_results(
            femap, prediction_type="edgewise", rank_metric="MUE", num_bootstraps=NBOOTSTRAP,
        )
        comparison_ddg = _reorient_comparison(comparison_ddg, "presto", "default")
        summary_ddg.insert(0, "target", target)
        comparison_ddg.insert(0, "target", target)
        print("  DDG (edgewise, MUE):")
        print(summary_ddg.to_string(index=False))
        print(comparison_ddg.to_string(index=False))
        all_summary_ddg.append(summary_ddg)
        all_comparison_ddg.append(comparison_ddg)

        summary_dg, comparison_dg = compare_and_rank_results(
            femap, prediction_type="nodewise", rank_metric="KTAU", num_bootstraps=NBOOTSTRAP,
        )
        comparison_dg = _reorient_comparison(comparison_dg, "MLE(presto)", "MLE(default)")
        summary_dg.insert(0, "target", target)
        comparison_dg.insert(0, "target", target)
        print("  DG (nodewise, KTAU):")
        print(summary_dg.to_string(index=False))
        print(comparison_dg.to_string(index=False))
        all_summary_dg.append(summary_dg)
        all_comparison_dg.append(comparison_dg)

    pd.concat(all_summary_ddg).to_csv(results_dir / f"paired_comparison_ddg_summary{suffix}.csv", index=False)
    pd.concat(all_comparison_ddg).to_csv(results_dir / f"paired_comparison_ddg{suffix}.csv", index=False)
    pd.concat(all_summary_dg).to_csv(results_dir / f"paired_comparison_dg_summary{suffix}.csv", index=False)
    pd.concat(all_comparison_dg).to_csv(results_dir / f"paired_comparison_dg{suffix}.csv", index=False)


# ---------------------------------------------------------------------------
# Section 5: Wilcoxon signed-rank + sign tests
# ---------------------------------------------------------------------------


def _run_wilcoxon_sign(
    abs_err_presto: np.ndarray, abs_err_default: np.ndarray, label: str
) -> dict:
    diffs = abs_err_presto - abs_err_default
    nonzero = diffs[diffs != 0]

    if len(nonzero) < 2:
        return {
            "label": label, "n": len(diffs),
            "n_presto_better": int(np.sum(diffs < 0)),
            "n_default_better": int(np.sum(diffs > 0)),
            "wilcoxon_stat": np.nan, "wilcoxon_p": np.nan, "sign_test_p": np.nan,
        }

    w_stat, w_p = wilcoxon(abs_err_presto, abs_err_default)
    n_presto_better = int(np.sum(diffs < 0))
    n_total = int(np.sum(diffs != 0))
    sign_p = binomtest(n_presto_better, n_total, 0.5).pvalue

    return {
        "label": label, "n": len(diffs),
        "n_presto_better": n_presto_better,
        "n_default_better": int(np.sum(diffs > 0)),
        "wilcoxon_stat": w_stat, "wilcoxon_p": w_p, "sign_test_p": sign_p,
    }


def run_statistical_tests(
    femaps: dict[str, dict[str, FEMap]],
    results_dir: Path,
    targets: list[str] | None = None,
    suffix: str = "",
) -> pd.DataFrame:
    if targets is None:
        targets = TARGETS

    rows = []
    pooled_ddg_presto, pooled_ddg_default = [], []
    pooled_dg_presto, pooled_dg_default = [], []

    for target in targets:
        graph_p = femaps[target]["presto"].to_legacy_graph()
        graph_d = femaps[target]["default"].to_legacy_graph()

        edge_map_p = {tuple(sorted([u, v])): d for u, v, d in graph_p.edges(data=True)}
        edge_map_d = {tuple(sorted([u, v])): d for u, v, d in graph_d.edges(data=True)}
        shared_edges = sorted(set(edge_map_p) & set(edge_map_d))
        abs_err_p = np.array([abs(edge_map_p[k]["calc_DDG"] - edge_map_p[k]["exp_DDG"]) for k in shared_edges])
        abs_err_d = np.array([abs(edge_map_d[k]["calc_DDG"] - edge_map_d[k]["exp_DDG"]) for k in shared_edges])

        rows.append(_run_wilcoxon_sign(abs_err_p, abs_err_d, f"{target}_DDG"))
        pooled_ddg_presto.append(abs_err_p)
        pooled_ddg_default.append(abs_err_d)

        node_p = extract_node_data(graph_p)
        node_d = extract_node_data(graph_d)
        dg_map_p = dict(zip(node_p["labels"], zip(node_p["exp_dg"], node_p["calc_dg"])))
        dg_map_d = dict(zip(node_d["labels"], zip(node_d["exp_dg"], node_d["calc_dg"])))
        shared_ligs = sorted(set(dg_map_p) & set(dg_map_d))
        abs_err_dg_p = np.array([abs(dg_map_p[l][1] - dg_map_p[l][0]) for l in shared_ligs])
        abs_err_dg_d = np.array([abs(dg_map_d[l][1] - dg_map_d[l][0]) for l in shared_ligs])

        rows.append(_run_wilcoxon_sign(abs_err_dg_p, abs_err_dg_d, f"{target}_DG"))
        pooled_dg_presto.append(abs_err_dg_p)
        pooled_dg_default.append(abs_err_dg_d)

    rows.append(_run_wilcoxon_sign(np.concatenate(pooled_ddg_presto), np.concatenate(pooled_ddg_default), "pooled_DDG"))
    rows.append(_run_wilcoxon_sign(np.concatenate(pooled_dg_presto), np.concatenate(pooled_dg_default), "pooled_DG"))

    df = pd.DataFrame(rows)
    fname = f"statistical_tests{suffix}.csv"
    df.to_csv(results_dir / fname, index=False)
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Section 6: Cyclopropyl-carbonyl substructure analysis
# ---------------------------------------------------------------------------


def find_cyclopropyl_ligands(smiles_map: dict[str, str]) -> set[str]:
    pattern = Chem.MolFromSmarts(CYCLOPROPYL_CARBONYL_SMARTS)
    hits = set()
    for name, smi in smiles_map.items():
        mol = Chem.MolFromSmiles(smi)
        if mol and mol.HasSubstructMatch(pattern):
            hits.add(name)
    return hits


def filter_cross_edges(
    df: pd.DataFrame, substructure_ligands: set[str], orient_towards: bool = False,
) -> pd.DataFrame:
    """Filter to edges where exactly one endpoint has the substructure.

    If orient_towards is True, flip edges so that Ligand2 is always the
    substructure-containing ligand (i.e. the transformation goes *to* it).
    """
    has_a = df["Ligand1"].isin(substructure_ligands)
    has_b = df["Ligand2"].isin(substructure_ligands)
    out = df[has_a != has_b].copy()

    if orient_towards:
        needs_flip = out["Ligand1"].isin(substructure_ligands)
        out.loc[needs_flip, ["Ligand1", "Ligand2"]] = (
            out.loc[needs_flip, ["Ligand2", "Ligand1"]].values
        )
        out.loc[needs_flip, "mean_calc_DDG"] *= -1
        out.loc[needs_flip, "exp_DDG"] *= -1

    return out


def run_cyclopropyl_analysis(
    predictions: dict[str, dict[str, pd.DataFrame]],
    experimentals: dict[str, pd.DataFrame],
    smiles_maps: dict[str, dict[str, str]],
    results_dir: Path,
) -> None:
    for target in TARGETS:
        cp_ligands = find_cyclopropyl_ligands(smiles_maps[target])
        if not cp_ligands:
            continue

        print(f"\n  --- {target.upper()} ---")
        print(f"  Cyclopropyl-carbonyl ligands: {sorted(cp_ligands)}")

        cross_preds = {}
        for ff in FORCE_FIELDS:
            cross_preds[ff] = filter_cross_edges(
                predictions[target][ff], cp_ligands, orient_towards=True,
            )
            print(f"  {FF_DISPLAY[ff]}: {len(cross_preds[ff])} cross-edges")

        n_edges = len(cross_preds[FORCE_FIELDS[0]])
        if n_edges == 0:
            print("  No cross-edges found, skipping.")
            continue

        # Build FEMaps for the cross-edges only (no MLE — graph is disconnected)
        exp_df = experimentals[target]
        femaps_cp: dict[str, FEMap] = {}
        for ff in FORCE_FIELDS:
            femaps_cp[ff] = build_femap(cross_preds[ff], exp_df, source=ff, run_mle=False)

        # Compute shared axis limits across both force fields
        all_vals = []
        for ff in FORCE_FIELDS:
            graph = femaps_cp[ff].to_legacy_graph()
            ed = extract_edge_data(graph)
            all_vals.extend(ed["exp_ddg"])
            all_vals.extend(ed["calc_ddg"])
        pad = 0.5
        shared_lim = (min(all_vals) - pad, max(all_vals) + pad)

        # DDG plots (per FF)
        for ff in FORCE_FIELDS:
            graph = femaps_cp[ff].to_legacy_graph()
            edge_data = extract_edge_data(graph)
            fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE))
            plot_correlation(
                ax, edge_data["exp_ddg"], edge_data["calc_ddg"],
                xerr=np.zeros_like(edge_data["exp_ddg"]),
                yerr=edge_data["calc_dddg"],
                ff_name=FF_DISPLAY[ff],
                target_name=f"{target.upper()} cyclopropyl cross",
                quantity="ΔΔG",
                statistics=DDG_STATISTICS,
                bootstrap_y_uncertainty=True,
                xy_lim=shared_lim,
            )
            fig.tight_layout()
            out = results_dir / f"{target}_{ff}_ddg_cyclopropyl.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out.name}")

        # Panel plot (1 row × 2 cols: default left, presto right)
        panel_w = FIGSIZE * len(FORCE_FIELDS) * 1.3
        fig_panel, axes_panel = plt.subplots(
            1, len(FORCE_FIELDS),
            figsize=(panel_w, FIGSIZE),
            squeeze=False,
        )
        for col, ff in enumerate(PANEL_FF_ORDER):
            ax = axes_panel[0, col]
            graph = femaps_cp[ff].to_legacy_graph()
            edge_data = extract_edge_data(graph)
            plot_correlation(
                ax, edge_data["exp_ddg"], edge_data["calc_ddg"],
                xerr=np.zeros_like(edge_data["exp_ddg"]),
                yerr=edge_data["calc_dddg"],
                ff_name=FF_DISPLAY[ff],
                target_name=f"{target.upper()} cyclopropyl cross",
                quantity="ΔΔG",
                statistics=DDG_STATISTICS,
                bootstrap_y_uncertainty=True,
                xy_lim=shared_lim,
            )
        fig_panel.subplots_adjust(wspace=0.4, hspace=0.7)
        out = results_dir / f"panel_{target}_ddg_cyclopropyl.png"
        fig_panel.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig_panel)
        print(f"  Saved {out.name}")

        # Interactive DDG plots (per FF)
        for ff in FORCE_FIELDS:
            graph = femaps_cp[ff].to_legacy_graph()
            edge_data = extract_edge_data(graph)
            title = f"ΔΔG {target.upper()} cyclopropyl cross — {FF_DISPLAY[ff]}"
            fig_ddg = _make_ddg_figure(edge_data, smiles_maps[target], title)
            fig_ddg.legend.location = "top_left"
            fig_ddg.legend.click_policy = "hide"
            out = results_dir / f"{target}_{ff}_ddg_cyclopropyl_interactive.html"
            bokeh.io.save(fig_ddg, filename=str(out), title=title)
            print(f"  Saved {out.name}")

        # Overlay interactive plot
        ff_colors = {"presto": "steelblue", "default": "darkorange"}
        fig_overlay = None
        for ff in FORCE_FIELDS:
            graph = femaps_cp[ff].to_legacy_graph()
            edge_data = extract_edge_data(graph)
            fig_overlay = _make_ddg_figure(
                edge_data, smiles_maps[target],
                f"ΔΔG {target.upper()} cyclopropyl cross — overlay",
                color=ff_colors[ff], legend_label=FF_DISPLAY[ff], figure=fig_overlay,
            )
        fig_overlay.legend.location = "top_left"
        fig_overlay.legend.click_policy = "hide"
        out = results_dir / f"{target}_overlay_ddg_cyclopropyl_interactive.html"
        bokeh.io.save(fig_overlay, filename=str(out), title=f"DDG cyclopropyl overlay {target.upper()}")
        print(f"  Saved {out.name}")

        # Bootstrap statistics
        print(f"\n  Bootstrap statistics (cross-edges, n={NBOOTSTRAP}):")
        for ff in FORCE_FIELDS:
            graph = femaps_cp[ff].to_legacy_graph()
            edge_data = extract_edge_data(graph)
            for metric in DDG_STATISTICS:
                s = bootstrap_statistic(
                    edge_data["exp_ddg"], edge_data["calc_ddg"],
                    dy_pred=edge_data["calc_dddg"],
                    statistic=metric, nbootstrap=NBOOTSTRAP, include_pred_uncertainty=True,
                )
                print(f"    {FF_DISPLAY[ff]:>14}  {metric}: {s['mle']:.2f} [95%: {s['low']:.2f}, {s['high']:.2f}]")

        # Per-edge comparison table
        print(f"\n  Per-edge errors:")
        print(f"    {'Edge':<30} {'Exp DDG':>8} {'Presto':>8} {'OpenFF':>8} {'|Err| P':>8} {'|Err| O':>8}")
        graph_p = femaps_cp["presto"].to_legacy_graph()
        graph_d = femaps_cp["default"].to_legacy_graph()
        edge_map_p = {tuple(sorted([u, v])): d for u, v, d in graph_p.edges(data=True)}
        edge_map_d = {tuple(sorted([u, v])): d for u, v, d in graph_d.edges(data=True)}
        shared = sorted(set(edge_map_p) & set(edge_map_d))

        abs_err_p_list, abs_err_d_list = [], []
        for k in shared:
            dp, dd = edge_map_p[k], edge_map_d[k]
            err_p = abs(dp["calc_DDG"] - dp["exp_DDG"])
            err_d = abs(dd["calc_DDG"] - dd["exp_DDG"])
            abs_err_p_list.append(err_p)
            abs_err_d_list.append(err_d)
            label = f"{k[0]} — {k[1]}"
            cp_mark = lambda n: "*" if n in cp_ligands else " "
            print(f"    {label:<30} {dp['exp_DDG']:>+8.2f} {dp['calc_DDG']:>+8.2f} {dd['calc_DDG']:>+8.2f} {err_p:>8.2f} {err_d:>8.2f}")

        abs_err_p = np.array(abs_err_p_list)
        abs_err_d = np.array(abs_err_d_list)

        # Wilcoxon / sign test
        print(f"\n  Statistical tests (cross-edges):")
        result = _run_wilcoxon_sign(abs_err_p, abs_err_d, f"{target}_cyclopropyl_DDG")
        for key, val in result.items():
            print(f"    {key}: {val}")

        # Paired bootstrap comparison (edgewise only — no MLE available)
        try:
            combined = build_combined_femap(cross_preds, exp_df)
            summary, comparison = compare_and_rank_results(
                combined, prediction_type="edgewise", rank_metric="MUE", num_bootstraps=NBOOTSTRAP,
            )
            comparison = _reorient_comparison(comparison, "presto", "default")
            summary.insert(0, "target", target)
            comparison.insert(0, "target", target)
            print(f"\n  Paired bootstrap comparison (MUE, presto - OpenFF 1.3.1):")
            print(summary.to_string(index=False))
            print(comparison.to_string(index=False))
            summary.to_csv(results_dir / f"paired_comparison_ddg_cyclopropyl_summary.csv", index=False)
            comparison.to_csv(results_dir / f"paired_comparison_ddg_cyclopropyl.csv", index=False)
            print(f"  Saved paired_comparison_ddg_cyclopropyl.csv")
            print(f"  Saved paired_comparison_ddg_cyclopropyl_summary.csv")
        except Exception as e:
            print(f"\n  Paired bootstrap comparison failed: {e}")

        # Save statistical tests to CSV
        result_df = pd.DataFrame([result])
        result_df.to_csv(results_dir / f"statistical_tests_cyclopropyl.csv", index=False)
        print(f"  Saved statistical_tests_cyclopropyl.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def analyse_rbfe(raw_data_dir: Path, ligands_dir: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    smiles_maps = {t: load_smiles(t, ligands_dir) for t in TARGETS}
    predictions = {
        t: {ff: load_predictions(t, ff, raw_data_dir) for ff in FORCE_FIELDS}
        for t in TARGETS
    }
    experimentals = {t: load_experimental(t, raw_data_dir) for t in TARGETS}

    print("Building FEMaps...")
    femaps: dict[str, dict[str, FEMap]] = {}
    for t in TARGETS:
        femaps[t] = {}
        for ff in FORCE_FIELDS:
            femaps[t][ff] = build_femap(predictions[t][ff], experimentals[t], source=ff)

    combined_femaps: dict[str, FEMap] = {}
    for t in TARGETS:
        combined_femaps[t] = build_combined_femap(predictions[t], experimentals[t])

    print("Building FEMaps with exclusions...")
    predictions_excl = {}
    experimentals_excl = {}
    for t in TARGETS:
        excl = EXCLUDE_LIGANDS.get(t, [])
        predictions_excl[t] = {ff: filter_predictions(predictions[t][ff], excl) for ff in FORCE_FIELDS}
        experimentals_excl[t] = filter_experimental(experimentals[t], excl)

    femaps_excl: dict[str, dict[str, FEMap]] = {}
    for t in TARGETS:
        femaps_excl[t] = {}
        for ff in FORCE_FIELDS:
            femaps_excl[t][ff] = build_femap(predictions_excl[t][ff], experimentals_excl[t], source=ff)

    combined_femaps_excl: dict[str, FEMap] = {}
    for t in TARGETS:
        combined_femaps_excl[t] = build_combined_femap(predictions_excl[t], experimentals_excl[t])

    print("\n" + "=" * 60)
    print("Section 1: Correlation Plots")
    print("=" * 60)
    run_cinnabar_plots(femaps, results_dir)
    run_cinnabar_plots(femaps_excl, results_dir, targets=["tyk2"], suffix="_no_ejm44")

    print("\n" + "=" * 60)
    print("Section 1b: Panel Plots")
    print("=" * 60)
    run_panel_plots(femaps, results_dir, femaps_no_excl=femaps_excl)

    print("\n" + "=" * 60)
    print("Section 2: Bootstrap Statistics (n=10000)")
    print("=" * 60)
    run_bootstrap_statistics(femaps, results_dir)
    run_bootstrap_statistics(femaps_excl, results_dir, targets=["tyk2"], suffix="_no_ejm44")

    print("\n" + "=" * 60)
    print("Section 3: Interactive Plots with 2D Structure Hover")
    print("=" * 60)
    run_interactive_plots(femaps, smiles_maps, results_dir)
    run_interactive_plots(femaps_excl, smiles_maps, results_dir, targets=["tyk2"], suffix="_no_ejm44")

    print("\n" + "=" * 60)
    print("Section 4: Paired Bootstrap Comparison (presto vs default)")
    print("=" * 60)
    run_paired_comparison(combined_femaps, results_dir)
    run_paired_comparison(combined_femaps_excl, results_dir, targets=["tyk2"], suffix="_no_ejm44")

    print("\n" + "=" * 60)
    print("Section 5: Wilcoxon and Sign Tests")
    print("=" * 60)
    run_statistical_tests(femaps, results_dir)
    run_statistical_tests(femaps_excl, results_dir, targets=["tyk2"], suffix="_no_ejm44")

    print("\n" + "=" * 60)
    print("Section 6: Cyclopropyl-Carbonyl Cross-Edge Analysis")
    print("=" * 60)
    run_cyclopropyl_analysis(predictions, experimentals, smiles_maps, results_dir)

    print("\n" + "=" * 60)
    print(f"All results saved to {results_dir}")
    print("=" * 60)

