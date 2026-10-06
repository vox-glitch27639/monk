#!/usr/bin/env python3
"""Bioimpedance analysis with synthetic Cole–Cole data and grouped model validation."""
import argparse
import json
import sys
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = "Monk"
OUTPUT_DIR = Path(__file__).resolve().parent / "result_output"
CSV_PATH = OUTPUT_DIR / "chicken_impedance.csv"
MODEL_PATH = OUTPUT_DIR / "monk_model.joblib"
MAX_HOURS = 48.0
RNG_SEED = 42
CAT_EDGES = [0, 16, 32, MAX_HOURS + 0.01]
CAT_LABELS = ["Fresh", "Moderate", "Spoiled"]
FREQ_POINTS = np.arange(10000, 100001, 10000)
NUM_SAMPLES = 12
TIME_POINTS = [0, 2, 4, 8, 12, 16, 20, 24, 30, 36, 42, 48]
R_EXTRA_0, R_INTRA_0, C_MEM_0, ALPHA_0 = 120.0, 350.0, 8e-9, 0.72
dR_EXT_H, dR_INT_H, dC_MEM_H, dALPHA_H = -0.30, -0.15, -0.09e-9, 0.003
TEMP_MEAN, TEMP_STD, TEMP_COEFF = 25.0, 2.5, -0.005
NOISE_RE_STD, NOISE_IM_STD = 3.5, 3.0
VAR_RE, VAR_RI, VAR_CM, VAR_AL = 15.0, 30.0, 0.8e-9, 0.03
REQUIRED_COLS = {"sample_id", "temperature_c", "freq_hz", "magnitude", "phase"}


def ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cole_cole_impedance(freq, R_ext, R_int, C_mem, alpha):
    """Cole–Cole impedance, with relaxation time R_int * C_mem."""
    omega = 2.0 * np.pi * freq
    tau = R_int * C_mem
    Z_int = R_int / (1.0 + (1j * omega * tau) ** alpha)
    return R_ext + Z_int


def spoiled_params(hours, temp_c, offsets):
    R_ext = R_EXTRA_0 + offsets['dRe'] + dR_EXT_H * hours
    R_int = R_INTRA_0 + offsets['dRi'] + dR_INT_H * hours
    C_mem = C_MEM_0   + offsets['dCm'] + dC_MEM_H * hours
    alpha = ALPHA_0   + offsets['dAl'] + dALPHA_H * hours

    # Simulated resistance increases at lower temperatures.
    tf = 1.0 + TEMP_COEFF * (temp_c - TEMP_MEAN)
    R_ext *= tf
    R_int *= tf

    R_ext = max(R_ext, 5.0)
    R_int = max(R_int, 10.0)
    C_mem = max(C_mem, 0.1e-9)
    alpha = np.clip(alpha, 0.5, 0.99)

    return R_ext, R_int, C_mem, alpha


def generate_dataset(out_path=CSV_PATH):
    """Generate synthetic chicken bioimpedance CSV."""
    out_path = ensure_parent(out_path)
    rng = np.random.default_rng(RNG_SEED)
    rows = []

    for idx in range(1, NUM_SAMPLES + 1):
        sid = f"S{idx:02d}"

        # Keep specimen offsets fixed across storage times.
        offsets = {
            'dRe': rng.normal(0, VAR_RE),
            'dRi': rng.normal(0, VAR_RI),
            'dCm': rng.normal(0, VAR_CM),
            'dAl': rng.normal(0, VAR_AL),
        }

        for hours in TIME_POINTS:
            temp = rng.normal(TEMP_MEAN, TEMP_STD)
            Re, Ri, Cm, al = spoiled_params(hours, temp, offsets)
            Z = cole_cole_impedance(FREQ_POINTS, Re, Ri, Cm, al)

            Z_noisy = (Z.real + rng.normal(0, NOISE_RE_STD, len(FREQ_POINTS))
                       + 1j * (Z.imag + rng.normal(0, NOISE_IM_STD, len(FREQ_POINTS))))

            mag = np.abs(Z_noisy)
            phase = np.degrees(np.arctan2(Z_noisy.imag, Z_noisy.real))

            for f, re, im, m, p in zip(FREQ_POINTS, Z_noisy.real, Z_noisy.imag, mag, phase):
                rows.append({
                    'sample_id': sid,
                    'sweep_id': f'{sid}-{hours}h',
                    'data_source': 'synthetic',
                    'elapsed_hours': hours,
                    'temperature_c': round(temp, 2),
                    'freq_hz': int(f),
                    'real': round(re, 4),
                    'imag': round(im, 4),
                    'magnitude': round(m, 4),
                    'phase': round(p, 4),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    trend = (df.groupby('elapsed_hours')['magnitude']
               .mean().reset_index()
               .rename(columns={'magnitude': 'mean_|Z|'}))

    print(f"\n  Synthetic dataset saved → {out_path}")
    print(f"      Rows: {len(df)}  |  Samples: {NUM_SAMPLES}"
          f"  |  Time pts: {len(TIME_POINTS)}  |  Freq pts: {len(FREQ_POINTS)}")
    print(f"\n  Spoilage trend (mean |Z| should rise over time):")
    for _, r in trend.iterrows():
        bar = '█' * int(r['mean_|Z|'] / 10)
        print(f"    {int(r['elapsed_hours']):3d} h  │  {r['mean_|Z|']:7.1f} Ω  {bar}")
    print()
    return df


def validate_data(df, training=True):
    """Validate each complete sweep; never silently average duplicate readings."""
    missing = REQUIRED_COLS - set(df.columns)
    if training and "elapsed_hours" not in df:
        missing.add("elapsed_hours")
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("CSV has no measurements.")
    df = df.copy()
    for col in ("sample_id", "sweep_id"):
        if col in df:
            if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
                raise ValueError(f"{col} must not contain empty identifiers.")
            df[col] = df[col].astype(str)
    numeric = ["temperature_c", "freq_hz", "magnitude", "phase"]
    numeric += [c for c in ("elapsed_hours", "real", "imag") if c in df]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{col} contains missing or non-finite values.")
    if (df.freq_hz <= 0).any() or (df.freq_hz % 1 != 0).any():
        raise ValueError("freq_hz must contain positive integer frequencies in Hz.")
    if (df.magnitude <= 0).any():
        raise ValueError("magnitude must be positive and calibrated in ohms.")
    if (df.phase.abs() > 180).any():
        raise ValueError("phase must be in degrees within [-180, 180].")
    if "elapsed_hours" in df and (df.elapsed_hours < 0).any():
        raise ValueError("elapsed_hours must be nonnegative.")
    keys = sweep_keys(df)
    if df.duplicated(keys + ["freq_hz"]).any():
        raise ValueError("Duplicate frequency within a sweep. Give repeated sweeps distinct sweep_id values.")
    grouped = df.groupby(keys, sort=True)
    if grouped.temperature_c.nunique().gt(1).any():
        raise ValueError("temperature_c must be constant within each sweep; record one sample temperature per sweep.")
    if "elapsed_hours" in df and grouped.elapsed_hours.nunique().gt(1).any():
        raise ValueError("elapsed_hours must be constant within each sweep.")
    frequencies = sorted(df.freq_hz.unique())
    if len(frequencies) < 2:
        raise ValueError("At least two different frequencies are needed per sweep.")
    if grouped.freq_hz.nunique().ne(len(frequencies)).any():
        raise ValueError("Incomplete sweep: all sweeps must contain the same frequencies.")
    return df


def sweep_keys(df):
    if "sweep_id" in df:
        return ["sample_id", "sweep_id"]
    if "elapsed_hours" in df:
        return ["sample_id", "elapsed_hours"]
    return ["sample_id"]


def load_data(path, training=True):
    df = validate_data(pd.read_csv(path), training=training)
    print(f"Loaded {len(df)} readings from {path} ({df.sample_id.nunique()} specimens).")
    return df


def engineer_features(df, training=True):
    """One row per sweep, with vectorized spectral features and no target leakage."""
    df = validate_data(df, training=training)
    keys = sweep_keys(df)
    pivot = df.pivot(index=keys, columns="freq_hz", values=["magnitude", "phase"])
    frequencies = sorted(df.freq_hz.unique())
    mag = pivot["magnitude"].reindex(columns=frequencies).to_numpy(dtype=float)
    phase = pivot["phase"].reindex(columns=frequencies).to_numpy(dtype=float)
    mag_cols = [f"mag_{int(f)}Hz" for f in frequencies]
    phase_cols = [f"phase_{int(f)}Hz" for f in frequencies]
    # Fit log-magnitude slopes across the frequency axis.
    log_f = np.log10(frequencies)
    centered_f = log_f - log_f.mean()
    slopes = np.log10(mag) @ centered_f / (centered_f @ centered_f)
    meta = df.groupby(keys, sort=True).first().reindex(pivot.index).reset_index()
    derived = np.column_stack((slopes, mag.mean(axis=1), phase.mean(axis=1),
                               np.ptp(mag, axis=1), np.ptp(phase, axis=1),
                               meta.temperature_c.to_numpy()))
    feature_cols = mag_cols + phase_cols + ["mag_slope_loglog", "mean_magnitude",
        "mean_phase", "mag_range", "phase_range", "temperature_c"]
    X = pd.DataFrame(np.column_stack((mag, phase, derived)), columns=feature_cols)
    y = meta.elapsed_hours.copy() if "elapsed_hours" in meta else None
    groups = meta.sample_id.copy()
    meta_cols = keys + [c for c in ("elapsed_hours", "data_source") if c in meta and c not in keys]
    print(f"Features: {len(X)} sweeps × {len(feature_cols)} features.")
    return X, y, groups, meta[meta_cols].copy(), feature_cols


def build_models():
    return {
        'Monk-LR': Pipeline([
            ('scaler', StandardScaler()),
            ('model',  Ridge(alpha=1.0)),
        ]),
        'Monk-RF': Pipeline([
            ('model', RandomForestRegressor(
                n_estimators=300, max_depth=8,
                min_samples_leaf=2, random_state=RNG_SEED)),
        ]),
        'Monk-GB': Pipeline([
            ('model', GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.08,
                subsample=0.8, random_state=RNG_SEED)),
        ]),
    }


def to_freshness_score(hours, max_h=MAX_HOURS):
    """Illustrative age index: 100 at zero hours, 0 at max_h."""
    return np.clip(100.0 * (1.0 - np.asarray(hours) / max_h), 0, 100)


def to_category(hours, edges=None, labels=None):
    """Illustrative age bins; boundaries belong to the lower-age category."""
    edges = CAT_EDGES if edges is None else edges
    labels = CAT_LABELS if labels is None else labels
    h = np.clip(np.asarray(hours, dtype=float), edges[0], edges[-1])
    return pd.cut(h, bins=edges, labels=labels, include_lowest=True)


def train_evaluate(X, y, groups, model_names=None, jobs=1):
    """Compare models using specimen-grouped validation, plus a mean-age baseline."""
    if pd.Series(groups).nunique() < 2:
        raise ValueError("Training requires at least two independent sample_id groups.")
    logo = LeaveOneGroupOut()
    results = {}
    baseline = np.empty(len(y))
    for train, test in logo.split(X, y, groups):
        baseline[test] = np.asarray(y)[train].mean()
    print(f"Mean-age baseline MAE: {mean_absolute_error(y, baseline):.2f} h")
    for name, pipe in build_models().items():
        if model_names and name not in model_names:
            continue
        predicted = cross_val_predict(pipe, X, y, cv=logo, groups=groups, n_jobs=jobs)
        results[name] = {"mae": float(mean_absolute_error(y, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
            "r2": float(r2_score(y, predicted)), "y_pred_cv": predicted,
            "baseline_mae": float(mean_absolute_error(y, baseline))}
        print(f"{name}: MAE {results[name]['mae']:.2f} h | RMSE {results[name]['rmse']:.2f} h | R² {results[name]['r2']:.4f}")
    if not results:
        raise ValueError("No valid models selected.")
    best = min(results, key=lambda name: results[name]["mae"])
    return results, best


def save_model(X, y, best_name, feature_cols, path=MODEL_PATH, data_source="unspecified"):
    pipe = build_models()[best_name]
    pipe.fit(X, y)
    frequencies = [int(c.removeprefix("mag_").removesuffix("Hz"))
                   for c in feature_cols if c.startswith("mag_") and c.endswith("Hz")]
    payload = {"project": PROJECT, "schema_version": 2,
        "model_name": best_name, "pipeline": pipe, "feature_cols": feature_cols,
        "frequencies": frequencies, "max_hours": MAX_HOURS,
        "cat_edges": CAT_EDGES, "cat_labels": CAT_LABELS, "data_source": data_source}
    joblib.dump(payload, ensure_parent(path))
    print(f"Saved {best_name} to {path}")
    return pipe


def load_model(path=MODEL_PATH):
    # joblib uses pickle: load only your own or otherwise trusted model files.
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("Unsupported model format. Train a model with monk.py --full.")
    required = {"pipeline", "feature_cols", "frequencies", "max_hours", "cat_edges", "cat_labels", "model_name", "data_source"}
    if required - payload.keys():
        raise ValueError("Saved model metadata is incomplete; retrain the model.")
    if not np.isfinite(payload["max_hours"]) or payload["max_hours"] <= 0:
        raise ValueError("Saved model max_hours must be positive and finite.")
    return payload


def freshness_table(y, results, best_name, meta):
    y_pred = results[best_name]['y_pred_cv']

    tbl = meta.copy()
    tbl['true_h'] = y.values
    tbl['pred_h'] = np.round(y_pred, 1)
    tbl['true_score'] = np.round(to_freshness_score(y.values), 1)
    tbl['pred_score'] = np.round(to_freshness_score(y_pred), 1)
    tbl['true_cat'] = to_category(y.values)
    tbl['pred_cat'] = to_category(y_pred)

    print(f"\n  Freshness predictions ({best_name}, LOGO-CV, first 20 sweeps):")
    print("  " + tbl.head(20).to_string(index=False))
    return tbl


def plot_all(df_raw, results, best_name, y, feature_cols, best_pipe):
    """Save the five impedance and model comparison plots."""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    fig1, ax1 = plt.subplots(figsize=(10, 5.5))
    freqs = sorted(df_raw['freq_hz'].unique())
    colours = cm.viridis(np.linspace(0.1, 0.9, len(freqs)))

    for freq, col in zip(freqs, colours):
        sub = df_raw[df_raw['freq_hz'] == freq]
        grp = sub.groupby('elapsed_hours')['magnitude'].agg(['mean', 'std']).reset_index()
        ax1.plot(grp['elapsed_hours'], grp['mean'],
                 marker='o', ms=5, color=col, label=f'{int(freq/1000)} kHz')
        ax1.fill_between(grp['elapsed_hours'],
                         grp['mean'] - grp['std'],
                         grp['mean'] + grp['std'],
                         alpha=0.12, color=col)

    ax1.set_xlabel('Elapsed Time (hours)', fontsize=12)
    ax1.set_ylabel('Impedance Magnitude |Z| (Ω)', fontsize=12)
    ax1.set_title(f'{PROJECT} — Bioimpedance |Z| vs Chicken Tissue Age\n'
                  '(mean ± 1σ across replicates)', fontsize=11)
    ax1.legend(title='Frequency', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(f'{OUTPUT_DIR}/fig1_magnitude_vs_time.png', dpi=150)
    print("  fig1_magnitude_vs_time.png")

    fig2, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5), squeeze=False)
    model_names = list(results.keys())

    for ax, name in zip(axes.ravel(), model_names):
        yp = results[name]['y_pred_cv']
        mae = results[name]['mae']
        r2 = results[name]['r2']
        is_best = (name == best_name)

        ax.scatter(y, yp, alpha=0.65, s=55, edgecolors='k', linewidths=0.4,
                   color='mediumseagreen' if is_best else 'steelblue')
        lims = [min(y.min(), yp.min()) - 2, max(y.max(), yp.max()) + 2]
        ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('True Hours', fontsize=10)
        ax.set_ylabel('Predicted Hours', fontsize=10)
        best_label = ' (best)' if is_best else ''
        ax.set_title(f'{name}{best_label}\nMAE={mae:.2f}h  R²={r2:.3f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig2.suptitle(f'{PROJECT} — Predicted vs Actual Elapsed Hours (LOGO-CV)',
                  fontsize=13, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(f'{OUTPUT_DIR}/fig2_pred_vs_actual.png', dpi=150)
    print("  fig2_pred_vs_actual.png")

    fig3, ax3 = plt.subplots(figsize=(10, 5.5))
    colors_m = {'Monk-LR': 'royalblue',
                'Monk-RF': 'darkorange',
                'Monk-GB': 'forestgreen'}
    markers_m = {'Monk-LR': 'o', 'Monk-RF': 's', 'Monk-GB': 'D'}

    for name, res in results.items():
        score = to_freshness_score(res['y_pred_cv'])
        ax3.scatter(y, score, label=name, marker=markers_m[name], s=50,
                    color=colors_m[name], alpha=0.75, edgecolors='k', linewidths=0.3)

    t_ref = np.linspace(0, MAX_HOURS, 200)
    ax3.plot(t_ref, to_freshness_score(t_ref), 'k--', lw=1.5, label='Ideal')
    ax3.axhline(67, color='green', ls=':', lw=1, alpha=0.6, label='Fresh threshold')
    ax3.axhline(33, color='red',   ls=':', lw=1, alpha=0.6, label='Spoiled threshold')
    ax3.set_xlabel('True Elapsed Hours', fontsize=12)
    ax3.set_ylabel('Predicted Freshness Score (0–100)', fontsize=12)
    ax3.set_title(f'{PROJECT} — Freshness Score vs Tissue Age (LOGO-CV)', fontsize=11)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.set_ylim(-5, 110)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(f'{OUTPUT_DIR}/fig3_freshness_score.png', dpi=150)
    print("  fig3_freshness_score.png")

    fig4, ax4 = plt.subplots(figsize=(8, 7))

    model_step = best_pipe.named_steps.get('model')
    if hasattr(model_step, 'feature_importances_'):
        importances = model_step.feature_importances_
        title_extra = '(impurity-based importance)'
    elif hasattr(model_step, 'coef_'):
        importances = np.abs(model_step.coef_)
        title_extra = '(|coefficient| on scaled features)'
    else:
        importances = np.zeros(len(feature_cols))
        title_extra = ''

    order = np.argsort(importances)
    ax4.barh([feature_cols[i] for i in order], importances[order],
             color='teal', edgecolor='k', linewidth=0.4)
    ax4.set_xlabel('Importance', fontsize=11)
    ax4.set_title(f'{PROJECT} — {best_name} Feature Importance\n{title_extra}',
                  fontsize=11)
    ax4.grid(True, axis='x', alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(f'{OUTPUT_DIR}/fig4_feature_importance.png', dpi=150)
    print("  fig4_feature_importance.png")

    fig5, ax5 = plt.subplots(figsize=(6, 5))

    y_true_cat = to_category(y.values)
    y_pred_cat = to_category(results[best_name]['y_pred_cv'])
    cm_matrix = confusion_matrix(y_true_cat, y_pred_cat, labels=CAT_LABELS)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_matrix,
                                  display_labels=CAT_LABELS)
    disp.plot(ax=ax5, cmap='Blues', colorbar=False)
    ax5.set_title(f'{PROJECT} — {best_name}\nFreshness Classification (LOGO-CV)',
                  fontsize=11)
    fig5.tight_layout()
    fig5.savefig(f'{OUTPUT_DIR}/fig5_confusion_matrix.png', dpi=150)
    print("  fig5_confusion_matrix.png")

    plt.close('all')


def predict_from_csv(csv_path, model_path=MODEL_PATH):
    payload = load_model(model_path)
    df = load_data(csv_path, training=False)
    frequencies = sorted(df.freq_hz.unique())
    if frequencies != payload["frequencies"]:
        raise ValueError(f"Frequency mismatch. Expected Hz: {payload['frequencies']}; received: {frequencies}")
    X, _, _, meta, _ = engineer_features(df, training=False)
    predicted = payload["pipeline"].predict(X[payload["feature_cols"]])
    out = meta.copy()
    out["pred_hours"] = np.round(predicted, 1)
    out["freshness"] = np.round(to_freshness_score(predicted, payload["max_hours"]), 1)
    out["category"] = to_category(predicted, payload["cat_edges"], payload["cat_labels"])
    out["model_training_source"] = payload["data_source"]
    out_csv = ensure_parent(OUTPUT_DIR / f"{Path(csv_path).stem}_predictions.csv")
    out.to_csv(out_csv, index=False)
    generate_html_report(out, payload["model_name"], OUTPUT_DIR / "monk_report.html")
    print(out.to_string(index=False))
    print(f"Predictions: {out_csv}")
    return out


def generate_demo_data():
    """Generate synthetic specimens at 3, 22 and 44 hours."""
    rng = np.random.default_rng(99)
    rows = []

    demo_samples = [
        ('DEMO_FRESH',     3,  24.5),
        ('DEMO_MODERATE', 22,  26.0),
        ('DEMO_SPOILED',  44,  25.2),
    ]

    for sid, hours, temp in demo_samples:
        offsets = {
            'dRe': rng.normal(0, VAR_RE * 0.5),
            'dRi': rng.normal(0, VAR_RI * 0.5),
            'dCm': rng.normal(0, VAR_CM * 0.5),
            'dAl': rng.normal(0, VAR_AL * 0.5),
        }

        Re, Ri, Cm, al = spoiled_params(hours, temp, offsets)
        Z = cole_cole_impedance(FREQ_POINTS, Re, Ri, Cm, al)

        Z_noisy = (Z.real + rng.normal(0, NOISE_RE_STD, len(FREQ_POINTS))
                   + 1j * (Z.imag + rng.normal(0, NOISE_IM_STD, len(FREQ_POINTS))))

        mag = np.abs(Z_noisy)
        phase = np.degrees(np.arctan2(Z_noisy.imag, Z_noisy.real))

        for f, re, im, m, p in zip(FREQ_POINTS, Z_noisy.real, Z_noisy.imag, mag, phase):
            rows.append({
                'sample_id': sid,
                'sweep_id': f'{sid}-demo',
                'data_source': 'synthetic',
                'elapsed_hours': hours,
                'temperature_c': round(temp, 2),
                'freq_hz': int(f),
                'real': round(re, 4),
                'imag': round(im, 4),
                'magnitude': round(m, 4),
                'phase': round(p, 4),
            })

    df = pd.DataFrame(rows)
    demo_csv = f"{OUTPUT_DIR}/demo_test_data.csv"
    df.to_csv(ensure_parent(demo_csv), index=False)
    print(f"\n  Demo test data saved → {demo_csv}")
    print(f"      3 fake samples: DEMO_FRESH (3h), DEMO_MODERATE (22h), DEMO_SPOILED (44h)")
    return demo_csv


def _color_for_score(score):
    if score >= 67:
        return '#22c55e'
    elif score >= 33:
        return '#f59e0b'
    else:
        return '#ef4444'


def _emoji_for_cat(cat):
    return {'Fresh': '🟢', 'Moderate': '🟡', 'Spoiled': '🔴'}.get(str(cat), '⚪')


def _bg_for_cat(cat):
    return {
        'Fresh':    'rgba(34,197,94,0.08)',
        'Moderate': 'rgba(245,158,11,0.08)',
        'Spoiled':  'rgba(239,68,68,0.08)',
    }.get(str(cat), 'transparent')


def generate_html_report(results_df, model_name, output_path):
    """Write an escaped HTML report with age-index limitations."""
    from datetime import datetime

    n_total = len(results_df)
    avg_score = results_df['freshness'].mean()
    n_fresh = (results_df['category'] == 'Fresh').sum()
    n_moderate = (results_df['category'] == 'Moderate').sum()
    n_spoiled = (results_df['category'] == 'Spoiled').sum()

    table_rows = ""
    for _, row in results_df.iterrows():
        s = row['freshness']
        cat = str(row['category'])
        bar_color = _color_for_score(s)
        bg = _bg_for_cat(cat)
        table_rows += f"""
        <tr style="background:{bg}">
          <td><strong>{escape(str(row['sample_id']))}</strong></td>
          <td>{row.get('elapsed_hours','—')}</td>
          <td>{row['pred_hours']}</td>
          <td>
            <div class="bar-wrap">
              <div class="bar" style="width:{max(s,2)}%; background:{bar_color}"></div>
              <span class="bar-label">{s:.0f}</span>
            </div>
          </td>
          <td>{_emoji_for_cat(cat)} {cat}</td>
        </tr>"""

    cards_html = ""
    for sid in results_df['sample_id'].unique():
        sub = results_df[results_df['sample_id'] == sid]
        avg_s = sub['freshness'].mean()
        cat = sub['category'].mode().iloc[0] if len(sub) > 0 else 'Unknown'
        color = _color_for_score(avg_s)
        emoji = _emoji_for_cat(cat)
        cards_html += f"""
        <div class="card">
          <div class="card-emoji">{emoji}</div>
          <div class="card-id">{escape(str(sid))}</div>
          <div class="card-score" style="color:{color}">{avg_s:.0f}</div>
          <div class="card-label">Freshness Score</div>
          <div class="card-cat" style="background:{color}20; color:{color};
               border:1px solid {color}40; border-radius:12px; padding:2px 12px;
               font-size:0.85rem; font-weight:600; margin-top:6px; display:inline-block">
            {cat}
          </div>
        </div>"""

    provenance = ", ".join(sorted(set(results_df.get("model_training_source", pd.Series(["unspecified"])).astype(str))))
    avg_color = _color_for_score(avg_score)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Monk — Freshness Report</title>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --surface2: #334155;
    --text: #f1f5f9; --text2: #94a3b8; --accent: #38bdf8;
    --green: #22c55e; --amber: #f59e0b; --red: #ef4444;
    --radius: 12px;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text); line-height:1.6;
    min-height:100vh;
  }}
  .header {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1a1a2e 100%);
    border-bottom: 1px solid var(--surface2);
    padding: 2rem 2rem 1.5rem;
    text-align: center;
  }}
  .header h1 {{
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.3rem;
  }}
  .header p {{ color: var(--text2); font-size: 0.95rem; }}
  .header .model-badge {{
    display: inline-block; background: var(--surface2); color: var(--accent);
    padding: 4px 14px; border-radius: 20px; font-size: 0.8rem;
    font-weight: 600; margin-top: 8px; letter-spacing: 0.5px;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 1.5rem; }}

  .summary {{
    display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap;
  }}
  .stat {{
    flex:1; min-width:130px; background: var(--surface);
    border-radius: var(--radius); padding: 1.2rem; text-align:center;
    border: 1px solid var(--surface2);
  }}
  .stat .num {{
    font-size: 2rem; font-weight: 800; line-height:1;
  }}
  .stat .lbl {{ font-size:0.8rem; color:var(--text2); margin-top:4px; }}

  .cards {{ display:flex; gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap; }}
  .card {{
    flex:1; min-width:160px; background:var(--surface);
    border-radius:var(--radius); padding:1.2rem; text-align:center;
    border:1px solid var(--surface2);
    transition: transform 0.2s, box-shadow 0.2s;
  }}
  .card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
  }}
  .card-emoji {{ font-size:1.8rem; margin-bottom:4px; }}
  .card-id {{ font-weight:700; font-size:1rem; color:var(--text); }}
  .card-score {{ font-size:2.2rem; font-weight:800; line-height:1.1; margin:4px 0; }}
  .card-label {{ font-size:0.75rem; color:var(--text2); }}

  .table-wrap {{
    background: var(--surface); border-radius: var(--radius);
    border: 1px solid var(--surface2); overflow: hidden;
  }}
  .table-title {{
    padding: 1rem 1.2rem 0.6rem; font-size:1.1rem; font-weight:700;
    color: var(--text);
  }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{
    background: var(--surface2); color: var(--text2);
    padding: 10px 14px; text-align:left; font-size:0.8rem;
    font-weight:600; text-transform:uppercase; letter-spacing:0.5px;
  }}
  td {{ padding: 10px 14px; border-top: 1px solid var(--surface2); font-size:0.9rem; }}
  tr:hover {{ background: rgba(56,189,248,0.04) !important; }}

  .bar-wrap {{
    display:flex; align-items:center; gap:8px;
  }}
  .bar {{
    height:10px; border-radius:5px; min-width:4px;
    transition: width 0.6s ease;
  }}
  .bar-label {{ font-weight:700; font-size:0.85rem; min-width:28px; }}

  .footer {{
    text-align:center; padding:2rem; color:var(--text2);
    font-size:0.8rem; border-top:1px solid var(--surface2);
    margin-top:2rem;
  }}
  .footer a {{ color:var(--accent); text-decoration:none; }}

  .legend {{
    display:flex; gap:1.5rem; justify-content:center;
    margin-bottom:1.5rem; font-size:0.85rem; color:var(--text2);
  }}
  .legend span {{
    display:inline-flex; align-items:center; gap:5px;
  }}
  .dot {{
    width:10px; height:10px; border-radius:50%; display:inline-block;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>🍗 Monk</h1>
  <p>Bioimpedance-Based Chicken Tissue Freshness Report</p>
  <div class="model-badge">Model: {escape(str(model_name))}</div>
</div>

<div class="container">

  <p style="padding:16px; margin-bottom:20px; border:1px solid var(--surface2); border-radius:8px">
    Research prototype · Model training source: {escape(provenance)}.<br>
    Score and categories are age-based illustrations, not validated freshness or food-safety assessments.
    Raw AD5933 register values require calibration before use.
  </p>
  <div class="legend">
    <span><span class="dot" style="background:var(--green)"></span> Fresh (67–100)</span>
    <span><span class="dot" style="background:var(--amber)"></span> Moderate (33–67)</span>
    <span><span class="dot" style="background:var(--red)"></span> Spoiled (0–33)</span>
  </div>

  <div class="summary">
    <div class="stat">
      <div class="num" style="color:var(--accent)">{n_total}</div>
      <div class="lbl">Total Sweeps</div>
    </div>
    <div class="stat">
      <div class="num" style="color:{avg_color}">{avg_score:.0f}</div>
      <div class="lbl">Avg Freshness</div>
    </div>
    <div class="stat">
      <div class="num" style="color:var(--green)">{n_fresh}</div>
      <div class="lbl">🟢 Fresh</div>
    </div>
    <div class="stat">
      <div class="num" style="color:var(--amber)">{n_moderate}</div>
      <div class="lbl">🟡 Moderate</div>
    </div>
    <div class="stat">
      <div class="num" style="color:var(--red)">{n_spoiled}</div>
      <div class="lbl">🔴 Spoiled</div>
    </div>
  </div>

  <div class="cards">
    {cards_html}
  </div>

  <div class="table-wrap">
    <div class="table-title">Detailed Predictions</div>
    <table>
      <thead>
        <tr>
          <th>Sample</th>
          <th>True Hours</th>
          <th>Predicted Hours</th>
          <th style="min-width:180px">Freshness Score</th>
          <th>Category</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </div>

</div>

<div class="footer">
  Monk &nbsp;·&nbsp; Generated {timestamp}
  &nbsp;·&nbsp; Bioimpedance Freshness Prediction System
</div>

</body>
</html>"""

    with ensure_parent(output_path).open('w', encoding='utf-8') as f:
        f.write(html)

    print(f"  HTML report saved → {output_path}")
    print(f"      Open in browser:  file://{Path(output_path).resolve()}")


def main(argv=None):
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Monk — bioimpedance age-estimation research prototype")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--generate", action="store_true", help="Generate synthetic CSV only")
    modes.add_argument("--train", action="store_true", help="Train from an existing CSV")
    modes.add_argument("--predict", metavar="FILE", help="Predict from calibrated measurements; known age optional")
    modes.add_argument("--demo", action="store_true", help="Generate three synthetic test specimens and predict")
    modes.add_argument("--full", action="store_true", help="Generate synthetic data and train")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--model", type=Path, default=None, help="Trusted model file; defaults to output-dir/monk_model.joblib")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", choices=list(build_models()), default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel CV workers; -1 uses all cores")
    parser.add_argument("--no-plots", action="store_true", help="Skip the five charts and importing Matplotlib")
    args = parser.parse_args(argv)
    if args.jobs == 0 or args.jobs < -1:
        parser.error("--jobs must be -1 or a positive integer")
    OUTPUT_DIR = args.output_dir.resolve()
    if args.csv == CSV_PATH:
        args.csv = OUTPUT_DIR / CSV_PATH.name
    model_path = args.model or OUTPUT_DIR / MODEL_PATH.name
    print(f"Monk · research prototype · age-based categories are illustrative")
    try:
        if args.predict:
            predict_from_csv(args.predict, model_path)
            return 0
        if args.demo:
            predict_from_csv(generate_demo_data(), model_path)
            return 0
        if args.generate or args.full:
            generate_dataset(args.csv)
        if args.generate:
            return 0
        df = load_data(args.csv)
        X, y, groups, meta, features = engineer_features(df)
        results, best = train_evaluate(X, y, groups, args.models, args.jobs)
        source = ", ".join(sorted(set(df.data_source.astype(str)))) if "data_source" in df else "unspecified"
        pipe = save_model(X, y, best, features, model_path, source)
        table = freshness_table(y, results, best, meta)
        table.to_csv(ensure_parent(OUTPUT_DIR / "cross_validation_predictions.csv"), index=False)
        metrics = {name: {k: v for k, v in res.items() if k != "y_pred_cv"} for name, res in results.items()}
        ensure_parent(OUTPUT_DIR / "metrics.json").write_text(json.dumps({"best_model": best, "data_source": source,
            "evaluation": "leave-one-specimen-out; model selection shares these folds", "models": metrics}, indent=2), encoding="utf-8")
        if not args.no_plots:
            plot_all(df, results, best, y, features, pipe)
        print(f"Best: {best} · MAE: {results[best]['mae']:.2f} h · outputs: {OUTPUT_DIR}")
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())

