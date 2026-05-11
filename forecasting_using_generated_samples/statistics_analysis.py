from tqdm import tqdm
import torch
from diff_Model_2D import SDE, VPSDE, sde_loss_fn, VESDE
from Models_2D import ScoreModel
from typing import Callable, Union, Tuple
from torch.nn import Module
from diff_Model_2D import pc_sampling, ReverseDiffusionPredictor, LangevinDynamicsCorrector, NoneCorrector
from diff_training_2D import train
from Dataset_Loader_2D import clear_extreme_data, clear_diff_data
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
from generate_new_samples_2D import generate_coldwave_samples, \
    generate_coldwave_only, generate_gaussian, generate_smote_with_noise, \
    generate_dirichlet_mixup, get_extreme_samples_tensor, get_all_samples_tensor

from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import pandas as pd


import numpy as np
import pandas as pd
import torch

from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

# =========================================================
# 0. User config
# =========================================================

TEST_COUNTRY_LIST = [
    'Belgium', 'Croatia', 'Denmark', 'Finland', 'France',
    'Germany', 'Hungary', 'Ireland', 'Italy',
    'Lithuania', 'Latvia', 'Netherlands', 'Norway',
    'Poland', 'Romania', 'Slovenia', 'Sweden', 'Switzerland'
]

WEATHER_TYPE = 'coldwave'
NUM_SAMPLES = 1200
DEVICE = 'cuda:2'

COUNTRYWISE_CSV_PATH = 'countrywise_generator_metrics_split.csv'
STAT_TABLE_CSV_PATH = 'paper_stat_table.csv'
EVENT_TABLE_CSV_PATH = 'paper_event_table.csv'

# =========================================================
# 1. Feature groups
# =========================================================

STAT_FEATURES = [
    "load_mean", "load_std", "load_min", "load_max",
    "load_q90", "load_q95", "load_q99", "load_sum",
    "temp_mean", "temp_std", "temp_min", "temp_max",
    "temp_q01", "temp_q05", "temp_q95", "temp_q99",
    "load_temp_corr"
]

EVENT_FEATURES = [
    "load_time_peak", "load_time_peak_hour",
    "load_ramp_mean_abs", "load_ramp_max_abs",
    "load_duration_above_q95", "load_excess_above_q95",
    "temp_event_extreme", "temp_peak_hour",
    "peak_hour_gap"
]

ALL_FEATURES = STAT_FEATURES + EVENT_FEATURES

# =========================================================
# 2. Basic utils
# =========================================================

def to_numpy(samples):
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    return samples.astype(np.float32)

def flatten_samples(samples):
    samples = to_numpy(samples)
    return samples.reshape(samples.shape[0], -1)

# =========================================================
# 3. Feature extraction
# =========================================================

def extract_event_features(samples, weather_type='coldwave'):
    samples = to_numpy(samples)
    coldwave = (weather_type == 'coldwave')

    feats = []

    for x in samples:
        load = x[0]   # (8, 24)
        temp = x[1]   # (8, 24)

        load_flat = load.reshape(-1)
        temp_flat = temp.reshape(-1)

        row = {}

        # -----------------------------
        # Statistical descriptors
        # -----------------------------
        row["load_mean"] = float(load_flat.mean())
        row["load_std"] = float(load_flat.std())
        row["load_min"] = float(load_flat.min())
        row["load_max"] = float(load_flat.max())
        row["load_q90"] = float(np.quantile(load_flat, 0.90))
        row["load_q95"] = float(np.quantile(load_flat, 0.95))
        row["load_q99"] = float(np.quantile(load_flat, 0.99))
        row["load_sum"] = float(load_flat.sum())

        row["temp_mean"] = float(temp_flat.mean())
        row["temp_std"] = float(temp_flat.std())
        row["temp_min"] = float(temp_flat.min())
        row["temp_max"] = float(temp_flat.max())
        row["temp_q01"] = float(np.quantile(temp_flat, 0.01))
        row["temp_q05"] = float(np.quantile(temp_flat, 0.05))
        row["temp_q95"] = float(np.quantile(temp_flat, 0.95))
        row["temp_q99"] = float(np.quantile(temp_flat, 0.99))

        if np.std(load_flat) > 1e-8 and np.std(temp_flat) > 1e-8:
            row["load_temp_corr"] = float(np.corrcoef(load_flat, temp_flat)[0, 1])
        else:
            row["load_temp_corr"] = 0.0

        # -----------------------------
        # Event-level descriptors
        # -----------------------------
        load_time_mean = load.mean(axis=0)   # (24,)
        temp_time_mean = temp.mean(axis=0)   # (24,)

        row["load_time_peak"] = float(load_time_mean.max())
        row["load_time_peak_hour"] = int(np.argmax(load_time_mean))

        load_diff = np.diff(load, axis=-1)
        row["load_ramp_mean_abs"] = float(np.mean(np.abs(load_diff)))
        row["load_ramp_max_abs"] = float(np.max(np.abs(load_diff)))

        load_thresh = np.quantile(load_flat, 0.95)
        row["load_duration_above_q95"] = int((load_flat > load_thresh).sum())
        row["load_excess_above_q95"] = float(np.maximum(load_flat - load_thresh, 0).sum())

        if coldwave:
            row["temp_event_extreme"] = float(temp_flat.min())
            row["temp_peak_hour"] = int(np.argmin(temp_time_mean))
            temp_ext_hour = int(np.argmin(temp_time_mean))
        else:
            row["temp_event_extreme"] = float(temp_flat.max())
            row["temp_peak_hour"] = int(np.argmax(temp_time_mean))
            temp_ext_hour = int(np.argmax(temp_time_mean))

        load_peak_hour = int(np.argmax(load_time_mean))
        row["peak_hour_gap"] = abs(load_peak_hour - temp_ext_hour)

        feats.append(row)

    return pd.DataFrame(feats)

# =========================================================
# 4. Diversity
# =========================================================

def pairwise_diversity(X):
    if X.shape[0] < 2:
        return np.nan
    D = pairwise_distances(X, metric='euclidean')
    tri = D[np.triu_indices(X.shape[0], k=1)]
    return float(tri.mean())

def compute_diversity_metric(syn_samples, weather_type='coldwave'):
    df_syn = extract_event_features(syn_samples, weather_type=weather_type)
    X_syn_feat = df_syn[ALL_FEATURES].fillna(0).values
    return pairwise_diversity(X_syn_feat)

# =========================================================
# 5. MMD / WD
# =========================================================

def median_heuristic_gamma(X, Y):
    Z = np.vstack([X, Y])
    if Z.shape[0] > 1000:
        idx = np.random.choice(Z.shape[0], 1000, replace=False)
        Z = Z[idx]

    D = pairwise_distances(Z, metric='euclidean')
    tri = D[np.triu_indices(D.shape[0], k=1)]
    pos = tri[tri > 0]

    if len(pos) == 0:
        med = 1.0
    else:
        med = np.median(pos)

    sigma = med if med > 1e-12 else 1.0
    gamma = 1.0 / (2.0 * sigma * sigma)
    return gamma

def compute_mmd_rbf(X, Y, gamma=None):
    if gamma is None:
        gamma = median_heuristic_gamma(X, Y)

    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)

    mmd = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return float(mmd)

def compute_wasserstein_featurewise(df_ref, df_syn, feature_cols):
    wd_values = []

    for col in feature_cols:
        a = df_ref[col].dropna().values
        b = df_syn[col].dropna().values

        if len(a) == 0 or len(b) == 0:
            wd = np.nan
        else:
            wd = wasserstein_distance(a, b)

        if not np.isnan(wd):
            wd_values.append(wd)

    return float(np.mean(wd_values)) if len(wd_values) > 0 else np.nan

def compute_feature_group_distances(df_syn, df_ref, feature_cols):
    X_syn = df_syn[feature_cols].fillna(0).values
    X_ref = df_ref[feature_cols].fillna(0).values

    mmd = compute_mmd_rbf(X_syn, X_ref)
    wd = compute_wasserstein_featurewise(df_ref, df_syn, feature_cols)

    return {
        "wd": wd,
        "mmd": mmd
    }

# =========================================================
# 6. Single-method evaluation
# =========================================================

def evaluate_single_generator(
    method_name,
    syn_samples,
    real_all_samples,
    real_extreme_samples,
    weather_type='coldwave'
):
    df_all = extract_event_features(real_all_samples, weather_type=weather_type)
    df_ext = extract_event_features(real_extreme_samples, weather_type=weather_type)
    df_syn = extract_event_features(syn_samples, weather_type=weather_type)

    diversity = compute_diversity_metric(
        syn_samples=syn_samples,
        weather_type=weather_type
    )

    stat_to_all = compute_feature_group_distances(df_syn, df_all, STAT_FEATURES)
    stat_to_ext = compute_feature_group_distances(df_syn, df_ext, STAT_FEATURES)

    event_to_all = compute_feature_group_distances(df_syn, df_all, EVENT_FEATURES)
    event_to_ext = compute_feature_group_distances(df_syn, df_ext, EVENT_FEATURES)

    summary = {
        "method": method_name,
        "diversity": diversity,

        "stat_wd_to_all": stat_to_all["wd"],
        "stat_mmd_to_all": stat_to_all["mmd"],
        "stat_wd_to_extreme": stat_to_ext["wd"],
        "stat_mmd_to_extreme": stat_to_ext["mmd"],

        "event_wd_to_all": event_to_all["wd"],
        "event_mmd_to_all": event_to_all["mmd"],
        "event_wd_to_extreme": event_to_ext["wd"],
        "event_mmd_to_extreme": event_to_ext["mmd"],
    }

    return summary

# =========================================================
# 7. Build generators
# =========================================================

def build_generator_funcs(country, num_samples=1200, weather_type='coldwave'):
    return {
        "proposed": lambda: generate_coldwave_samples(
            country=country,
            num_samples=num_samples,
            weather_type=weather_type
        ),
        "direct": lambda: generate_coldwave_only(
            country=country,
            num_samples=num_samples,
            weather_type=weather_type
        ),
        "smote_with_noise": lambda: generate_smote_with_noise(
            country=country,
            num_samples=num_samples,
            weather_type=weather_type,
            k=10,
            lam_max=0.3,
            noise_std=0.03,
            smooth_kernel=5
        ),
        "dirichlet_mixup": lambda: generate_dirichlet_mixup(
            country=country,
            num_samples=num_samples,
            weather_type=weather_type,
            k=4,
            alpha=0.5
        ),
        "gaussian": lambda: generate_gaussian(
            country=country,
            num_samples=num_samples,
            weather_type=weather_type
        ),
    }

# =========================================================
# 8. Countrywise evaluation
# =========================================================

def run_countrywise_evaluation(
    test_country_list,
    weather_type='coldwave',
    num_samples=1200,
    device='cuda:2',
    save_csv_path='countrywise_generator_metrics_split.csv'
):
    all_rows = []

    for country in test_country_list:
        print(f"\n========== Processing country: {country} ==========")

        try:
            real_extreme = get_extreme_samples_tensor(
                country=country,
                weather_type=weather_type,
                device=device
            )

            real_all = get_all_samples_tensor(
                country=country,
                weather_type=weather_type,
                device=device
            )

            print(f"[{country}] real_extreme shape = {tuple(real_extreme.shape)}")
            print(f"[{country}] real_all shape     = {tuple(real_all.shape)}")

            generator_funcs = build_generator_funcs(
                country=country,
                num_samples=num_samples,
                weather_type=weather_type
            )

            for method_name, gen_fn in generator_funcs.items():
                print(f"  -> Generating and evaluating method: {method_name}")

                try:
                    syn_samples = gen_fn()

                    summary = evaluate_single_generator(
                        method_name=method_name,
                        syn_samples=syn_samples,
                        real_all_samples=real_all,
                        real_extreme_samples=real_extreme,
                        weather_type=weather_type
                    )

                    summary["country"] = country
                    summary["weather_type"] = weather_type
                    summary["num_syn_samples"] = int(syn_samples.shape[0])
                    summary["num_real_all"] = int(real_all.shape[0])
                    summary["num_real_extreme"] = int(real_extreme.shape[0])

                    all_rows.append(summary)

                except Exception as e:
                    print(f"  [ERROR] method={method_name}, country={country}, error={e}")
                    all_rows.append({
                        "country": country,
                        "weather_type": weather_type,
                        "method": method_name,
                        "num_syn_samples": np.nan,
                        "num_real_all": int(real_all.shape[0]),
                        "num_real_extreme": int(real_extreme.shape[0]),
                        "diversity": np.nan,
                        "stat_wd_to_all": np.nan,
                        "stat_mmd_to_all": np.nan,
                        "stat_wd_to_extreme": np.nan,
                        "stat_mmd_to_extreme": np.nan,
                        "event_wd_to_all": np.nan,
                        "event_mmd_to_all": np.nan,
                        "event_wd_to_extreme": np.nan,
                        "event_mmd_to_extreme": np.nan,
                        "error": str(e)
                    })

        except Exception as e:
            print(f"[ERROR] country={country}, error={e}")
            for method_name in ["proposed", "direct", "smote_with_noise", "dirichlet_mixup", "gaussian"]:
                all_rows.append({
                    "country": country,
                    "weather_type": weather_type,
                    "method": method_name,
                    "num_syn_samples": np.nan,
                    "num_real_all": np.nan,
                    "num_real_extreme": np.nan,
                    "diversity": np.nan,
                    "stat_wd_to_all": np.nan,
                    "stat_mmd_to_all": np.nan,
                    "stat_wd_to_extreme": np.nan,
                    "stat_mmd_to_extreme": np.nan,
                    "event_wd_to_all": np.nan,
                    "event_mmd_to_all": np.nan,
                    "event_wd_to_extreme": np.nan,
                    "event_mmd_to_extreme": np.nan,
                    "error": str(e)
                })

    result_df = pd.DataFrame(all_rows)

    front_cols = [
        "country", "weather_type", "method",
        "num_syn_samples", "num_real_all", "num_real_extreme"
    ]
    other_cols = [c for c in result_df.columns if c not in front_cols]
    result_df = result_df[front_cols + other_cols]

    result_df.to_csv(save_csv_path, index=False)
    print(f"\nSaved countrywise results to: {save_csv_path}")

    return result_df

# =========================================================
# 9. Build compact paper tables
# =========================================================

def build_stat_table(result_df, save_path=None):
    target_methods = [
        "proposed",
        "direct",
        "smote_with_noise",
        "dirichlet_mixup",
        "gaussian"
    ]

    df = result_df[result_df["method"].isin(target_methods)].copy()

    table = (
        df.groupby("method", as_index=False)[
            [
                "diversity",
                "stat_wd_to_all",
                "stat_mmd_to_all",
                "stat_wd_to_extreme",
                "stat_mmd_to_extreme"
            ]
        ]
        .mean()
        .rename(columns={
            "method": "Method",
            "diversity": "Diversity ↑",
            "stat_wd_to_all": "WD to Real ↓",
            "stat_mmd_to_all": "MMD to Real ↓",
            "stat_wd_to_extreme": "WD to Extreme ↓",
            "stat_mmd_to_extreme": "MMD to Extreme ↓"
        })
    )

    method_order = ["proposed", "direct", "smote_with_noise", "dirichlet_mixup", "gaussian"]
    table["order"] = table["Method"].apply(lambda x: method_order.index(x))
    table = table.sort_values("order").drop(columns=["order"]).reset_index(drop=True)

    table.iloc[:, 1:] = table.iloc[:, 1:].round(4)

    if save_path is not None:
        table.to_csv(save_path, index=False)
        print(f"Saved statistical table to: {save_path}")

    return table

def build_event_table(result_df, save_path=None):
    target_methods = [
        "proposed",
        "direct",
        "smote_with_noise",
        "dirichlet_mixup",
        "gaussian"
    ]

    df = result_df[result_df["method"].isin(target_methods)].copy()

    table = (
        df.groupby("method", as_index=False)[
            [
                "diversity",
                "event_wd_to_all",
                "event_mmd_to_all",
                "event_wd_to_extreme",
                "event_mmd_to_extreme"
            ]
        ]
        .mean()
        .rename(columns={
            "method": "Method",
            "diversity": "Diversity ↑",
            "event_wd_to_all": "WD to Real ↓",
            "event_mmd_to_all": "MMD to Real ↓",
            "event_wd_to_extreme": "WD to Extreme ↓",
            "event_mmd_to_extreme": "MMD to Extreme ↓"
        })
    )

    method_order = ["proposed", "direct", "smote_with_noise", "dirichlet_mixup", "gaussian"]
    table["order"] = table["Method"].apply(lambda x: method_order.index(x))
    table = table.sort_values("order").drop(columns=["order"]).reset_index(drop=True)

    table.iloc[:, 1:] = table.iloc[:, 1:].round(4)

    if save_path is not None:
        table.to_csv(save_path, index=False)
        print(f"Saved event-level table to: {save_path}")

    return table

# =========================================================
# 10. Main
# =========================================================


result_df = run_countrywise_evaluation(
        test_country_list=TEST_COUNTRY_LIST,
        weather_type=WEATHER_TYPE,
        num_samples=NUM_SAMPLES,
        device=DEVICE,
        save_csv_path=COUNTRYWISE_CSV_PATH
    )

stat_table = build_stat_table(
        result_df,
        save_path=STAT_TABLE_CSV_PATH
    )

event_table = build_event_table(
        result_df,
        save_path=EVENT_TABLE_CSV_PATH
    )

print("\n========== Statistical descriptors table ==========")
print(stat_table)

print("\n========== Event-level descriptors table ==========")
print(event_table)