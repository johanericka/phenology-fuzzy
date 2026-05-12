"""
Visualization Module (FASE 8)
==============================
Generates all plots for simulation results analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def set_plot_style():
    """Set consistent plot style."""
    plt.rcParams.update({
        'figure.figsize': (14, 8),
        'font.size': 11,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
    })


def plot_season_timeseries(season_df: pd.DataFrame, title: str = "",
                           save_path: str = None):
    """
    Plot time-series for one season showing soil moisture, targets, and irrigation.
    """
    set_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    hst = season_df['hst'].values
    
    # --- Plot 1: Soil Moisture vs Targets ---
    ax = axes[0]
    ax.fill_between(hst, season_df['rL_pct'], season_df['rU_pct'],
                    alpha=0.2, color='green', label='Target Range')
    ax.plot(hst, season_df['r_pct'], 'b-', linewidth=1.5, label='Soil Moisture (%FC)')
    ax.plot(hst, season_df['rL_pct'], 'g--', linewidth=1, alpha=0.7, label='rL (Lower)')
    ax.plot(hst, season_df['rU_pct'], 'g-', linewidth=1, alpha=0.7, label='rU (Upper)')
    ax.plot(hst, season_df['rWP_pct'], 'r--', linewidth=1, label='Wilting Point')
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='Field Capacity')
    ax.set_ylabel('Soil Moisture (% FC)')
    ax.set_title(f'Soil Moisture Dynamics — {title}')
    ax.legend(loc='upper right', ncol=3)
    ax.set_ylim(0, 120)
    
    # Color zones for phases
    for _, row in season_df.iterrows():
        if row['phase'] == 'REPRODUKTIF':
            ax.axvspan(row['hst'] - 0.5, row['hst'] + 0.5,
                      alpha=0.05, color='red')
    
    # --- Plot 2: Irrigation & Precipitation ---
    ax = axes[1]
    ax.bar(hst, season_df['precipitation'], width=0.8, alpha=0.6,
           color='skyblue', label='Rainfall (mm)')
    ax.bar(hst, season_df['irrigation_mm'], width=0.4, alpha=0.8,
           color='orange', label='Irrigation (mm)')
    ax.set_ylabel('Water Input (mm)')
    ax.set_title('Rainfall & Irrigation Events')
    ax.legend()
    
    # --- Plot 3: Drought Indicator & Urgency ---
    ax = axes[2]
    ax.plot(hst, season_df['drought_indicator'], 'r-', linewidth=1,
            alpha=0.7, label='Drought Indicator (dᵢ)')
    ax.plot(hst, season_df['urgency'], 'orange', linewidth=1.5,
            label='Irrigation Urgency (uᵢ)')
    ax.fill_between(hst, season_df['urgency'], alpha=0.15, color='orange')
    ax.set_ylabel('Value (0-1)')
    ax.set_xlabel('Hari Setelah Tanam (HST)')
    ax.set_title('Fuzzy Control Signals')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_boxplots(fuzzy_summary: pd.DataFrame,
                             reactive_summary: pd.DataFrame,
                             save_path: str = None):
    """
    Boxplot comparison of key metrics between algorithms.
    """
    set_plot_style()
    metrics = {
        'iwu_mm': 'Irrigation Water Use (mm)',
        'mean_r_pct': 'Mean Soil Moisture (%FC)',
        'mse': 'MSE (tracking error)',
        'target_pct': 'Days in Target Range (%)',
        'stress_days': 'Stress Days',
        'n_irrigation_events': 'Irrigation Events',
    }
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (col, label) in enumerate(metrics.items()):
        ax = axes[idx]
        
        fuzzy_data = fuzzy_summary[col].dropna().values
        reactive_data = reactive_summary[col].dropna().values
        
        bp = ax.boxplot([fuzzy_data, reactive_data],
                       labels=['Fuzzy-Fenologi', 'Reaktif'],
                       patch_artist=True,
                       boxprops=dict(linewidth=1.5),
                       medianprops=dict(color='black', linewidth=2))
        
        bp['boxes'][0].set_facecolor('#4ECDC4')
        bp['boxes'][1].set_facecolor('#FF6B6B')
        
        ax.set_title(label)
        ax.set_ylabel(label.split('(')[0].strip())
    
    plt.suptitle('Perbandingan Performa: Fuzzy-Fenologi vs Reaktif', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_seasonal_comparison(fuzzy_summary: pd.DataFrame,
                             reactive_summary: pd.DataFrame,
                             save_path: str = None):
    """
    Bar plot comparing IWU and target accuracy per season.
    """
    set_plot_style()
    seasons = sorted(
        set(fuzzy_summary['season'].astype(str)).union(set(reactive_summary['season'].astype(str))),
        key=lambda s: (0 if 'Penghujan' in s else 1 if 'Peralihan' in s else 2 if 'Kemarau' in s else 99, s),
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, metric_col in enumerate(['iwu_mm', 'target_pct', 'stress_days']):
        ax = axes[idx]
        
        fuzzy_vals = [fuzzy_summary[fuzzy_summary['season'] == s][metric_col].mean()
                     for s in seasons]
        reactive_vals = [reactive_summary[reactive_summary['season'] == s][metric_col].mean()
                        for s in seasons]
        
        x = np.arange(len(seasons))
        width = 0.35
        
        ax.bar(x - width/2, fuzzy_vals, width, label='Fuzzy-Fenologi',
               color='#4ECDC4', edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, reactive_vals, width, label='Reaktif',
               color='#FF6B6B', edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(seasons)
        
        titles = {
            'iwu_mm': 'Irrigation Water Use (mm)',
            'target_pct': 'Days in Target Range (%)',
            'stress_days': 'Stress Days'
        }
        ax.set_title(titles.get(metric_col, metric_col))
        ax.legend()
    
    plt.suptitle('Perbandingan per Musim Tanam', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tradeoff(fuzzy_summary: pd.DataFrame,
                  reactive_summary: pd.DataFrame,
                  save_path: str = None):
    """
    Scatter plot: IWU vs Target Accuracy (trade-off analysis).
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(fuzzy_summary['iwu_mm'], fuzzy_summary['target_pct'],
              s=80, c='#4ECDC4', edgecolors='black', linewidth=0.5,
              label='Fuzzy-Fenologi', zorder=5)
    ax.scatter(reactive_summary['iwu_mm'], reactive_summary['target_pct'],
              s=80, c='#FF6B6B', edgecolors='black', linewidth=0.5,
              label='Reaktif', zorder=5)
    
    ax.set_xlabel('Irrigation Water Use (mm)')
    ax.set_ylabel('Days in Target Range (%)')
    ax.set_title('Trade-off Analysis: Efisiensi Air vs Akurasi Kontrol')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_method_soil_moisture_diagnostics(season_df: pd.DataFrame, title: str = "", save_path: str = None):
    """
    Plot terpisah untuk satu metode pada satu skenario:
    - Soil moisture vs target band
    - Deviasi terhadap target midpoint (over/under)
    - Precipitation
    - Irrigation
    """
    if season_df is None or season_df.empty:
        return

    set_plot_style()
    df = season_df.sort_values("hst").copy()
    h = df["hst"].values
    target_mid = (df["rL_pct"] + df["rU_pct"]) / 2.0
    deviation = df["r_pct"] - target_mid

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.1, 1.0, 1.0]})

    # 1) Soil moisture vs targets
    ax = axes[0]
    ax.fill_between(h, df["rL_pct"], df["rU_pct"], color="#77c26a", alpha=0.25, label="Rentang target")
    ax.plot(h, df["r_pct"], color="#114b8c", linewidth=2.0, label="Soil moisture")
    ax.plot(h, df["rL_pct"], "g--", linewidth=1.2, alpha=0.9, label="rL")
    ax.plot(h, df["rU_pct"], "g-", linewidth=1.2, alpha=0.9, label="rU")
    ax.plot(h, df["rWP_pct"], "r--", linewidth=1.1, alpha=0.8, label="rWP")
    ax.axhline(100, color="gray", linestyle=":", linewidth=1.0, label="Field Capacity")

    # Highlight over/under target conditions
    over_mask = df["r_pct"] > df["rU_pct"]
    under_mask = df["r_pct"] < df["rL_pct"]
    ax.fill_between(h, df["r_pct"], df["rU_pct"], where=over_mask, color="#f4a261", alpha=0.35, label="Over-irrigation / terlalu basah")
    ax.fill_between(h, df["r_pct"], df["rL_pct"], where=under_mask, color="#e76f51", alpha=0.25, label="Under-irrigation / terlalu kering")
    ax.set_ylabel("SM (%FC)")
    ax.set_title(f"Soil Moisture, Target, Hujan, dan Irigasi — {title}")
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, max(130, float(df[["r_pct", "rU_pct"]].max().max()) + 5))

    # 2) Deviation panel
    ax = axes[1]
    ax.axhline(0, color="black", linewidth=1.0)
    ax.plot(h, deviation, color="#5e548e", linewidth=1.6, label="SM - target_mid")
    ax.fill_between(h, 0, deviation, where=deviation >= 0, color="#f4a261", alpha=0.35, label="Di atas target midpoint")
    ax.fill_between(h, 0, deviation, where=deviation < 0, color="#457b9d", alpha=0.25, label="Di bawah target midpoint")
    ax.set_ylabel("Deviasi\n(%FC)")
    ax.legend(loc="upper right", ncol=2)

    # 3) Precipitation
    ax = axes[2]
    ax.bar(h, df["precipitation"], color="#4ea8de", alpha=0.85, width=0.8)
    ax.set_ylabel("Hujan\n(mm)")
    ax.set_title("Precipitation harian")

    # 4) Irrigation
    ax = axes[3]
    ax.bar(h, df["irrigation_mm"], color="#f77f00", alpha=0.9, width=0.8)
    ax.set_ylabel("Irigasi\n(mm)")
    ax.set_xlabel("Hari Setelah Tanam (HST)")
    ax.set_title("Irigasi harian (hasil controller)")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_controller_comparison_timeseries(
    fuzzy_df: pd.DataFrame,
    reactive_df: pd.DataFrame,
    title: str = "",
    save_path: str = None,
):
    """
    Plot one representative scenario comparing two controllers on:
    - soil moisture trajectories against the dynamic target band
    - daily rainfall
    - daily irrigation from both controllers

    The target band is taken from the fuzzy-phenology run because it is the
    proposed dynamic reference used for interpretation.
    """
    if fuzzy_df is None or reactive_df is None or fuzzy_df.empty or reactive_df.empty:
        return

    set_plot_style()
    fuzzy_df = fuzzy_df.sort_values("hst").copy()
    reactive_df = reactive_df.sort_values("hst").copy()

    h = fuzzy_df["hst"].values

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.0, 1.2]},
    )

    # 1) Soil moisture vs dynamic target band
    ax = axes[0]
    ax.fill_between(
        h,
        fuzzy_df["rL_pct"].values,
        fuzzy_df["rU_pct"].values,
        color="#8ecf7a",
        alpha=0.22,
        label="Dynamic target band (rL-rU)",
    )
    ax.plot(
        h,
        fuzzy_df["r_pct"].values,
        color="#0b6e4f",
        linewidth=2.2,
        label="Fuzzy-Phenology",
    )
    ax.plot(
        h,
        reactive_df["r_pct"].values,
        color="#c44536",
        linewidth=1.8,
        label="Reactive-Static",
    )
    ax.plot(h, fuzzy_df["rL_pct"].values, "--", color="#2a9d8f", linewidth=1.0, alpha=0.8)
    ax.plot(h, fuzzy_df["rU_pct"].values, "-", color="#2a9d8f", linewidth=1.0, alpha=0.8)
    ax.axhline(100, color="gray", linestyle=":", linewidth=1.0, alpha=0.8, label="Field capacity")
    ax.set_ylabel("Soil moisture (%FC)")
    ax.set_ylim(0, max(130, float(max(fuzzy_df["r_pct"].max(), reactive_df["r_pct"].max(), fuzzy_df["rU_pct"].max())) + 5))
    ax.set_title(f"Representative Time-Series Comparison — {title}")
    ax.legend(loc="upper right", ncol=2)

    # Shade reproductive period to show when stage sensitivity matters most.
    reproductive_mask = fuzzy_df["phase"].astype(str).eq("REPRODUKTIF").values
    if reproductive_mask.any():
        start_idx = None
        for idx, is_rep in enumerate(reproductive_mask):
            if is_rep and start_idx is None:
                start_idx = idx
            elif not is_rep and start_idx is not None:
                ax.axvspan(h[start_idx], h[idx - 1], color="#f4a261", alpha=0.08)
                start_idx = None
        if start_idx is not None:
            ax.axvspan(h[start_idx], h[-1], color="#f4a261", alpha=0.08)

    # 2) Rainfall
    ax = axes[1]
    ax.bar(h, fuzzy_df["precipitation"].values, color="#4ea8de", alpha=0.85, width=0.8)
    ax.set_ylabel("Rainfall\n(mm)")
    ax.set_title("Daily rainfall")

    # 3) Irrigation pulses from both controllers
    ax = axes[2]
    ax.bar(
        h - 0.18,
        fuzzy_df["irrigation_mm"].values,
        width=0.36,
        color="#0b6e4f",
        alpha=0.85,
        label="Fuzzy-Phenology",
    )
    ax.bar(
        h + 0.18,
        reactive_df["irrigation_mm"].values,
        width=0.36,
        color="#c44536",
        alpha=0.75,
        label="Reactive-Static",
    )
    ax.set_ylabel("Irrigation\n(mm)")
    ax.set_xlabel("Days after planting (DAP)")
    ax.set_title("Daily irrigation events")
    ax.legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_reproducibility_figure_composite_figure(
    representative_results: dict[str, pd.DataFrame],
    seasonal_summary_df: pd.DataFrame,
    overall_summary_df: pd.DataFrame,
    representative_title: str = "",
    save_path: str = None,
):
    """
    Build the representative scenario figure for the four-controller package.

    Panels:
    - dynamic target band, phenology phases, and soil-moisture trajectories for all four controllers
    - rainfall and controller irrigation inputs for the same scenario
    """
    if not representative_results:
        return
    if "fuzzy_phenology" not in representative_results:
        return

    colors = {
        "fuzzy_phenology": "#0b6e4f",
        "fuzzy_static": "#3c91e6",
        "reactive_static": "#c44536",
        "reactive_stage": "#9c6644",
    }
    labels = {
        "fuzzy_phenology": "Fuzzy-Phenology",
        "fuzzy_static": "Fuzzy-Static",
        "reactive_static": "Reactive-Static",
        "reactive_stage": "Reactive-Phenology",
    }

    set_plot_style()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 0.95], "hspace": 0.12},
    )
    fuzzy_df = representative_results["fuzzy_phenology"].sort_values("hst").copy()
    h = fuzzy_df["hst"].values
    display_title = representative_title.replace("_", " ")

    # Panel A: dynamic target band, phenology phases, and all controller trajectories.
    ax = axes[0]
    phase_colors = {
        "VEGETATIF": "#d8f3dc",
        "REPRODUKTIF": "#ffe8cc",
        "PEMATANGAN": "#e7f0ff",
    }
    phase_start = 0
    phases = fuzzy_df["phase"].astype(str).values
    for idx in range(1, len(phases) + 1):
        if idx == len(phases) or phases[idx] != phases[phase_start]:
            phase = phases[phase_start]
            ax.axvspan(
                h[phase_start],
                h[idx - 1],
                color=phase_colors.get(phase, "#eeeeee"),
                alpha=0.22,
                linewidth=0,
            )
            mid = (h[phase_start] + h[idx - 1]) / 2
            ax.text(mid, 7, phase.title(), ha="center", va="bottom", fontsize=8, color="#444444")
            phase_start = idx

    ax.fill_between(
        h,
        fuzzy_df["rL_pct"].values,
        fuzzy_df["rU_pct"].values,
        color="#8ecf7a",
        alpha=0.24,
        label="Dynamic target band",
    )
    for ctrl_id, df in representative_results.items():
        if df is None or df.empty:
            continue
        ordered = df.sort_values("hst")
        ax.plot(
            ordered["hst"].values,
            ordered["r_pct"].values,
            color=colors.get(ctrl_id, "#444444"),
            linewidth=2.0 if ctrl_id == "fuzzy_phenology" else 1.7,
            label=labels.get(ctrl_id, ctrl_id),
        )
    ax.plot(h, fuzzy_df["rL_pct"].values, "--", color="#2a9d8f", linewidth=1.1, alpha=0.9, label="Lower target")
    ax.plot(h, fuzzy_df["rU_pct"].values, "-", color="#2a9d8f", linewidth=1.1, alpha=0.9, label="Upper target")
    ax.axhline(100, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("Soil moisture (%FC)")
    ax.set_title(f"Dynamic target band, phenology phase, and controller trajectories — {display_title}")
    ax.set_ylim(0, max(130, float(max(df["r_pct"].max() for df in representative_results.values() if df is not None and not df.empty)) + 5))
    ax.legend(loc="upper right", ncol=3)

    # Panel B: rainfall and controller irrigation actions.
    ax = axes[1]
    ax.bar(h, fuzzy_df["precipitation"].values, color="#8ecae6", alpha=0.68, width=0.88, label="Rainfall")
    offsets = {
        "fuzzy_phenology": -0.27,
        "fuzzy_static": -0.09,
        "reactive_stage": 0.09,
        "reactive_static": 0.27,
    }
    for ctrl_id, df in representative_results.items():
        if df is None or df.empty:
            continue
        ordered = df.sort_values("hst")
        ax.bar(
            ordered["hst"].values + offsets.get(ctrl_id, 0.0),
            ordered["irrigation_mm"].values,
            color=colors.get(ctrl_id, "#444444"),
            alpha=0.78,
            width=0.16,
            label=f"{labels.get(ctrl_id, ctrl_id)} irrigation",
        )
    ax.set_ylabel("Water input (mm)")
    ax.set_xlabel("Days after planting (DAP)")
    ax.set_title("Daily rainfall and irrigation input")
    ax.legend(loc="upper right", ncol=3, fontsize=8)

    for axis in axes:
        axis.grid(True, alpha=0.28)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_reproducibility_figure_irrigation_summary_figure(
    seasonal_summary_df: pd.DataFrame,
    save_path: str = None,
):
    """
    Build the seasonal irrigation summary figure.

    Panels:
    - seasonal irrigation water use across the four controllers
    - seasonal irrigation action count across the four controllers
    """
    if seasonal_summary_df is None or seasonal_summary_df.empty:
        return

    colors = {
        "fuzzy_phenology": "#0b6e4f",
        "fuzzy_static": "#3c91e6",
        "reactive_static": "#c44536",
        "reactive_stage": "#9c6644",
    }
    labels = {
        "fuzzy_phenology": "Fuzzy-Phenology",
        "fuzzy_static": "Fuzzy-Static",
        "reactive_static": "Reactive-Static",
        "reactive_stage": "Reactive-Phenology",
    }

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    seasonal = seasonal_summary_df.copy()
    seasonal["season_order"] = seasonal["season"].map(
        lambda s: 0 if "Penghujan" in str(s) else 1 if "Peralihan" in str(s) else 2 if "Kemarau" in str(s) else 99
    )
    seasonal = seasonal.sort_values(["season_order", "controller_label"])
    seasons = seasonal["season"].drop_duplicates().tolist()
    season_labels = [str(season).replace("_", " ") for season in seasons]
    controller_order = [cid for cid in ["fuzzy_phenology", "fuzzy_static", "reactive_static", "reactive_stage"] if cid in seasonal["controller_id"].unique()]
    x = np.arange(len(seasons))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(controller_order))

    ax = axes[0]
    for offset, ctrl_id in zip(offsets, controller_order):
        ctrl_rows = seasonal[seasonal["controller_id"] == ctrl_id]
        values = []
        for season in seasons:
            row = ctrl_rows[ctrl_rows["season"] == season]
            values.append(float(row["iwu_mm"].iloc[0]) if not row.empty else np.nan)
        ax.bar(
            x + offset,
            values,
            width,
            label=labels.get(ctrl_id, ctrl_id),
            color=colors.get(ctrl_id, "#444444"),
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(season_labels, rotation=10)
    ax.set_ylabel("Mean IWU (mm)")
    ax.set_title("Seasonal irrigation water use")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    for offset, ctrl_id in zip(offsets, controller_order):
        ctrl_rows = seasonal[seasonal["controller_id"] == ctrl_id]
        values = []
        for season in seasons:
            row = ctrl_rows[ctrl_rows["season"] == season]
            values.append(float(row["n_irrigation_events"].iloc[0]) if not row.empty else np.nan)
        ax.bar(
            x + offset,
            values,
            width,
            label=labels.get(ctrl_id, ctrl_id),
            color=colors.get(ctrl_id, "#444444"),
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(season_labels, rotation=10)
    ax.set_ylabel("Mean irrigation actions (events season$^{-1}$)")
    ax.set_title("Seasonal irrigation action count")
    ax.legend(loc="upper left", fontsize=9)

    for axis in fig.axes:
        axis.grid(True, alpha=0.28)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()
