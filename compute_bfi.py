#!/usr/bin/env python3
"""
compute_bfi.py (CONFIGURABLE via bfi_config.yaml)

This module implements the BFI pipeline and includes a forensic-logger utility
that emits intermediate distributions and final metrics in JSON Lines (JSONL)
format for enterprise ingestion (SIEM / audit tooling).

Primary exported functions:
- load_config(path=None)
- load_dataset(path)
- per_stage_refusal_rates(df)
- intent_delta(df)
- compute_conditional_mutual_information(df) -> returns distributions and scalar quantities
- normalized_entropy(df)  # uses compute_conditional_mutual_information internally
- compute_lut_weight(df)
- stage_reliability_score(df)
- compute_hfi(intent_delta_mean, h_norm, lut_weight, srs_mean)
- compute_bfi_score(hfi_norm)
- analyze(df)
- write_forensic_log(df, out_path, run_id=None, extra=None)

CLI:
- --forensic-log <path> : write one JSONL line (a single JSON object) describing the run
"""

from typing import Tuple, Dict, Any
import os
import json
import math
import ast as _ast
from collections import Counter, defaultdict
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.utils import resample

# Config defaults — overridden by bfi_config.yaml if present
DEFAULT_CONFIG = {
    "bootstrap_samples": 10000,
    "random_seed": 42,
    "lut_mapping": {
        "(1, 1, 1)": 1.00,
        "(0, 1, 1)": 0.85,
        "(1, 0, 1)": 0.80,
        "(1, 1, 0)": 0.75,
        "(0, 0, 1)": 0.60,
        "(0, 1, 0)": 0.50,
        "(1, 0, 0)": 0.40,
        "(0, 0, 0)": 0.00,
    },
    "max_acceptable_lut": 0.15,
    "certification_threshold": 0.05,
    # divisor used to normalize HFI raw value into [0,1]; default equals theoretical upper bound 2.0
    "hfi_normalization_divisor": 2.0,
    "srs_transform_clip": 1.0,
}

def load_config(path: str = None) -> dict:
    """
    Load YAML config from path or env var BFI_CONFIG_PATH. Merge shallow keys with defaults.
    Non-fatal on failure: will print a warning and continue with defaults.
    """
    try:
        import yaml
    except Exception:
        print("[compute_bfi] WARNING: pyyaml not installed; using default config", file=os.sys.stderr)
        return dict(DEFAULT_CONFIG)

    if path is None:
        path = os.getenv("BFI_CONFIG_PATH", "bfi_config.yaml")
    cfg = dict(DEFAULT_CONFIG)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                file_cfg = yaml.safe_load(fh) or {}
            cfg.update(file_cfg)
    except Exception as e:
        print(f"[compute_bfi] WARNING: failed to load config {path}: {e}", file=os.sys.stderr)
    return cfg

# Module-level config (loaded at import)
CONFIG = load_config()

def _parse_lut_mapping(raw: Dict[str, float]) -> Dict[Tuple[int,int,int], float]:
    """
    Parse LUT mapping from config into a dict keyed by (lexical, ast, oracle) tuples.
    """
    parsed = {}
    for k, v in (raw or {}).items():
        try:
            if isinstance(k, str):
                tup = _ast.literal_eval(k)
                if isinstance(tup, tuple) and len(tup) == 3:
                    parsed[tuple(int(x) for x in tup)] = float(v)
                else:
                    parts = [int(p.strip()) for p in k.replace("(", "").replace(")","").split(",")]
                    if len(parts) == 3:
                        parsed[tuple(parts)] = float(v)
            elif isinstance(k, (tuple, list)) and len(k) == 3:
                parsed[tuple(int(x) for x in k)] = float(v)
        except Exception:
            continue
    for dk, dv in DEFAULT_CONFIG["lut_mapping"].items():
        tup = _ast.literal_eval(dk)
        if tup not in parsed:
            parsed[tup] = float(dv)
    return parsed

# ---------- Schema validation helpers ----------
def _to_binary_value(v):
    """
    Convert a scalar value to 0 or 1 if possible, else raise ValueError.
    Accepts integers 0/1, strings "0","1","true","false","t","f","yes","no" (case-insensitive).
    """
    if pd.isna(v):
        raise ValueError("missing value")
    if isinstance(v, (int, np.integer)):
        if int(v) in (0,1):
            return int(v)
        raise ValueError(f"integer value {v} not in (0,1)")
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return 1
    if s in ("0", "false", "f", "no", "n"):
        return 0
    raise ValueError(f"unrecognized binary value: {v!r}")

def _validate_and_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce schema:
      - Required columns: lexical, ast, oracle, label, topic
      - lexical/ast/oracle/label must be 0 or 1 (convertible)
      - topic must be present and non-null (string)
    Returns a new DataFrame with normalized dtypes:
      lexical/ast/oracle/label -> int (0/1)
      topic -> str
    Raises ValueError with descriptive messages if validation fails.
    """
    required = ["lexical", "ast", "oracle", "label", "topic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset schema error: missing required column(s): {missing}. Required columns: {required}")

    out = df.copy()

    binary_cols = ["lexical", "ast", "oracle", "label"]
    for col in binary_cols:
        bad_indices = []
        normalized = []
        for idx, v in out[col].iteritems():
            try:
                normalized.append(_to_binary_value(v))
            except ValueError as e:
                bad_indices.append((idx, v, str(e)))
                normalized.append(None)
        if bad_indices:
            examples = bad_indices[:5]
            msg_lines = [f"Column '{col}' contains non-binary or missing values. Examples (index, value, err):"]
            for idx, val, err in examples:
                msg_lines.append(f"  - index={idx!r}, value={val!r}, err={err}")
            msg_lines.append("All values in this column must be 0/1 (or convertible strings 'true'/'false').")
            raise ValueError("\n".join(msg_lines))
        out[col] = np.array(normalized, dtype=int)

    if out["topic"].isnull().any():
        idxs = out[out["topic"].isnull()].index.tolist()[:10]
        raise ValueError(f"Column 'topic' must not contain null values. Example bad indices: {idxs}")

    out["topic"] = out["topic"].astype(str).map(lambda s: s.strip())

    if (out["topic"] == "").any():
        idxs = out[out["topic"] == ""].index.tolist()[:10]
        raise ValueError(f"Column 'topic' contains empty strings. Example bad indices: {idxs}")

    return out

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load CSV from path and validate schema strictly.
    Returns DataFrame with columns:
      lexical, ast, oracle, label (int 0/1) and topic (str non-empty)
    Raises ValueError on any schema/type error (fail-fast).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    df = pd.read_csv(path, dtype=object)
    df_valid = _validate_and_normalize_columns(df)
    return df_valid

def per_stage_refusal_rates(df: pd.DataFrame) -> dict:
    rates = {}
    for stage in ("lexical", "ast", "oracle"):
        if stage in df.columns:
            rates[stage] = float(df[stage].mean())
        else:
            rates[stage] = 0.0
    return rates

def intent_delta(df: pd.DataFrame) -> Tuple[dict, float]:
    deltas = {}
    for stage in ("lexical", "ast", "oracle"):
        if stage not in df.columns or "label" not in df.columns:
            deltas[stage] = 0.0
            continue
        mask1 = df["label"] == 1
        mask0 = df["label"] == 0
        r1 = float(df[mask1][stage].mean()) if mask1.any() else 0.0
        r0 = float(df[mask0][stage].mean()) if mask0.any() else 0.0
        deltas[stage] = r1 - r0
    mean_delta = float(np.mean(list(deltas.values())))
    return deltas, mean_delta

# ---------- Conditional Mutual Information / H_norm (with intermediate distributions) ----------
def compute_conditional_mutual_information(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute and return the intermediate distributions and scalar quantities required to
    derive H_norm = 1 - ( I(R ; Y | T) / H(R) ).

    Returns a dict with keys:
      - N: int
      - p_r: { r_str: prob }
      - p_r_given_t: { topic: { r_str: prob } }
      - p_r_given_y_t: { f"{label}|{topic}": { r_str: prob } }
      - H_R: float
      - H_R_given_T: float
      - H_R_given_Y_T: float
      - I_R_Y_given_T: float
      - H_norm: float

    Where r_str is a string representation of the tuple, e.g. "(1, 0, 1)".
    """
    # Validate assumed columns exist (caller should have validated)
    for c in ("lexical","ast","oracle","label","topic"):
        if c not in df.columns:
            raise ValueError(f"compute_conditional_mutual_information: required column missing: {c}")

    res: Dict[str, Any] = {}
    N = len(df)
    res["N"] = int(N)
    if N == 0:
        # No data -> produce empty distributions and conservative H_norm=1.0
        res.update({
            "p_r": {},
            "p_r_given_t": {},
            "p_r_given_y_t": {},
            "H_R": 0.0,
            "H_R_given_T": 0.0,
            "H_R_given_Y_T": 0.0,
            "I_R_Y_given_T": 0.0,
            "H_norm": 1.0,
        })
        return res

    df2 = df.copy()
    df2["r"] = df2.apply(lambda row: (int(row["lexical"]), int(row["ast"]), int(row["oracle"])), axis=1)

    # Marginal p(r)
    counts_r = df2.groupby("r").size().to_dict()
    p_r = {str(k): float(v) / float(N) for k, v in counts_r.items()}
    res["p_r"] = dict(sorted(p_r.items()))

    # H(R)
    p_r_vals = np.array(list(counts_r.values()), dtype=float) / float(N) if counts_r else np.array([])
    H_R = float(entropy(p_r_vals, base=2)) if p_r_vals.size > 0 else 0.0
    res["H_R"] = float(H_R)

    # p(r | t)
    p_r_given_t = {}
    H_R_given_T = 0.0
    grouped_T = df2.groupby("topic")
    for t, group in grouped_T:
        n_t = len(group)
        counts_r_t = group.groupby("r").size().to_dict()
        total_t = sum(counts_r_t.values()) or 1
        p_map = {str(k): float(v) / float(total_t) for k, v in counts_r_t.items()}
        p_r_given_t[str(t)] = dict(sorted(p_map.items()))
        # compute H(R | T=t)
        p_vals = np.array(list(counts_r_t.values()), dtype=float) / float(total_t) if counts_r_t else np.array([])
        H_r_t = float(entropy(p_vals, base=2)) if p_vals.size > 0 else 0.0
        H_R_given_T += (n_t / float(N)) * H_r_t
    res["p_r_given_t"] = dict(sorted(p_r_given_t.items()))
    res["H_R_given_T"] = float(H_R_given_T)

    # p(r | y, t) and H(R | Y, T)
    p_r_given_y_t = {}
    H_R_given_Y_T = 0.0
    grouped_YT = df2.groupby(["label", "topic"])
    for (y, t), group in grouped_YT:
        n_yt = len(group)
        counts_r_yt = group.groupby("r").size().to_dict()
        total_yt = sum(counts_r_yt.values()) or 1
        key = f"label={int(y)}|topic={t}"
        p_map = {str(k): float(v) / float(total_yt) for k, v in counts_r_yt.items()}
        p_r_given_y_t[key] = dict(sorted(p_map.items()))
        p_vals = np.array(list(counts_r_yt.values()), dtype=float) / float(total_yt) if counts_r_yt else np.array([])
        H_r_yt = float(entropy(p_vals, base=2)) if p_vals.size > 0 else 0.0
        H_R_given_Y_T += (n_yt / float(N)) * H_r_yt
    res["p_r_given_y_t"] = dict(sorted(p_r_given_y_t.items()))
    res["H_R_given_Y_T"] = float(H_R_given_Y_T)

    # Conditional mutual information
    I_R_Y_given_T = H_R_given_T - H_R_given_Y_T
    if I_R_Y_given_T < 0 and I_R_Y_given_T > -1e-12:
        I_R_Y_given_T = 0.0
    I_R_Y_given_T = max(0.0, float(I_R_Y_given_T))
    res["I_R_Y_given_T"] = float(I_R_Y_given_T)

    # H_norm = 1 - (I / H_R) with guards
    if H_R <= 0.0:
        H_norm = 1.0
    else:
        H_norm = 1.0 - (I_R_Y_given_T / float(H_R))
        H_norm = float(max(0.0, min(1.0, H_norm)))
    res["H_norm"] = float(H_norm)

    return res

def normalized_entropy(df: pd.DataFrame) -> float:
    """
    Backwards-compatible wrapper that returns H_norm only.
    """
    return compute_conditional_mutual_information(df)["H_norm"]

def compute_lut_weight(df: pd.DataFrame) -> float:
    raw_lut = CONFIG.get("lut_mapping", DEFAULT_CONFIG["lut_mapping"])
    lut = _parse_lut_mapping(raw_lut)
    patterns = df.apply(lambda row: (int(row.lexical), int(row.ast), int(row.oracle)), axis=1)
    freq = Counter(patterns)
    total = sum(freq.values()) or 1
    weighted = sum(lut.get(pat, 0.0) * (count / total) for pat, count in freq.items())
    return float(weighted)

def stage_reliability_score(df: pd.DataFrame, n_boot: int = None, seed: int = None) -> dict:
    n_boot = int(n_boot if n_boot is not None else CONFIG.get("bootstrap_samples", 10000))
    seed = int(seed if seed is not None else CONFIG.get("random_seed", 42))
    n = len(df)
    if n == 0:
        return {"lexical": 0.0, "ast": 0.0, "oracle": 0.0, "mean": 0.0}

    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 2**31 - 1, size=n_boot)

    srs = {}
    for stage in ("lexical", "ast", "oracle"):
        rates = []
        values = df[stage].values
        for s in seeds:
            sample = resample(values, replace=True, n_samples=n, random_state=int(s))
            rates.append(np.mean(sample))
        std = float(np.std(rates))
        clip = float(CONFIG.get("srs_transform_clip", 1.0)) or 1.0
        transformed = 1.0 - min(clip, std * math.sqrt(max(1, n)))
        srs[stage] = float(max(0.0, min(1.0, transformed)))
    srs["mean"] = float(np.mean([srs["lexical"], srs["ast"], srs["oracle"]]))
    return srs

def compute_hfi(intent_delta_mean: float, h_norm: float, lut_weight: float, srs_mean: float) -> float:
    idelta_clamped = max(-1.0, min(1.0, float(intent_delta_mean)))
    raw = lut_weight * srs_mean * (1.0 - float(h_norm)) * (1.0 + idelta_clamped)
    divisor = float(CONFIG.get("hfi_normalization_divisor", 2.0)) or 2.0
    normalized = raw / divisor
    normalized = float(max(0.0, min(1.0, normalized)))
    return normalized

def compute_bfi_score(hfi_norm: float) -> float:
    return round(float(hfi_norm) * 100.0, 3)

def analyze(df: pd.DataFrame) -> dict:
    rates = per_stage_refusal_rates(df)
    deltas, mean_delta = intent_delta(df)
    # compute conditional MI and H_norm (and capture intermediate values if needed)
    cmi = compute_conditional_mutual_information(df)
    h_norm = cmi["H_norm"]
    lut_weight = compute_lut_weight(df)
    srs = stage_reliability_score(df)
    hfi = compute_hfi(mean_delta, h_norm, lut_weight, srs["mean"])
    bfi = compute_bfi_score(hfi)
    return {
        "per_stage_rates": rates,
        "intent_delta_per_stage": deltas,
        "intent_delta_mean": mean_delta,
        "conditional_mi": {
            "H_R": cmi["H_R"],
            "H_R_given_T": cmi["H_R_given_T"],
            "H_R_given_Y_T": cmi["H_R_given_Y_T"],
            "I_R_Y_given_T": cmi["I_R_Y_given_T"],
            "H_norm": cmi["H_norm"],
        },
        "lut_weight": lut_weight,
        "srs": srs,
        "hfi_normalized": hfi,
        "bfi": bfi,
    }

# ---------- Forensic logger ----------
def _tuple_key_to_str_map(d: Dict[tuple, float]) -> Dict[str, float]:
    return {str(k): float(v) for k, v in d.items()}

def write_forensic_log(df: pd.DataFrame, out_path: str, run_id: str = None, extra: Dict[str, Any] = None) -> None:
    """
    Write a single JSON object as one line in JSONL format to out_path (append mode).
    The object includes:
      - run_id: UUID4 string (generated if not provided)
      - timestamp: ISO8601 UTC
      - config snapshot (subset)
      - N
      - distributions: p_r, p_r_given_t, p_r_given_y_t (all mapping string -> float)
      - scalar metrics: H_R, H_R_given_T, H_R_given_Y_T, I_R_Y_given_T, H_norm
      - pipeline outputs: per_stage_rates, intent_delta_per_stage, intent_delta_mean, lut_weight, srs, hfi_normalized, bfi
      - extra: any additional metadata provided by caller
    The file is appended if exists; ensure proper file permissions for SIEM ingestion.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat() + "Z"

    # Ensure df validated (so distributions consistent)
    # We will not re-raise schema errors here; caller should have validated. But we validate to be safe.
    df_valid = _validate_and_normalize_columns(df)

    # Compute CMI distributions and scalars
    cmi = compute_conditional_mutual_information(df_valid)
    # Compute other pipeline outputs
    rates = per_stage_refusal_rates(df_valid)
    deltas, mean_delta = intent_delta(df_valid)
    lut_weight = compute_lut_weight(df_valid)
    srs = stage_reliability_score(df_valid)
    hfi = compute_hfi(mean_delta, cmi["H_norm"], lut_weight, srs["mean"])
    bfi = compute_bfi_score(hfi)

    # Build forensic payload (structured)
    payload = {
        "run_id": run_id,
        "timestamp": ts,
        "config": {
            "hfi_normalization_divisor": CONFIG.get("hfi_normalization_divisor"),
            "bootstrap_samples": CONFIG.get("bootstrap_samples"),
            "random_seed": CONFIG.get("random_seed"),
        },
        "dataset": {
            "N": cmi["N"],
        },
        "distributions": {
            "p_r": cmi.get("p_r", {}),
            "p_r_given_t": cmi.get("p_r_given_t", {}),
            "p_r_given_y_t": cmi.get("p_r_given_y_t", {}),
        },
        "conditional_mi": {
            "H_R": cmi.get("H_R"),
            "H_R_given_T": cmi.get("H_R_given_T"),
            "H_R_given_Y_T": cmi.get("H_R_given_Y_T"),
            "I_R_Y_given_T": cmi.get("I_R_Y_given_T"),
            "H_norm": cmi.get("H_norm"),
        },
        "pipeline": {
            "per_stage_rates": rates,
            "intent_delta_per_stage": deltas,
            "intent_delta_mean": mean_delta,
            "lut_weight": lut_weight,
            "srs": srs,
            "hfi_normalized": hfi,
            "bfi": bfi,
        },
        "extra": extra or {},
    }

    # Write single JSON object as one line (JSONL)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mode = "a"
    with open(out_path, mode, encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute BFI from dataset CSV.")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("--config", help="Path to bfi_config.yaml (overrides env var BFI_CONFIG_PATH)")
    parser.add_argument("--bootstrap", type=int, help="Override number of bootstrap samples")
    parser.add_argument("--forensic-log", help="Path to write forensic JSONL output (append mode)")
    args = parser.parse_args()
    if args.config:
        CONFIG.update(load_config(args.config))
    df = load_dataset(args.csv)
    if args.bootstrap:
        CONFIG["bootstrap_samples"] = args.bootstrap
    result = analyze(df)
    print(json.dumps(result, indent=2))
    if args.forensic_log:
        write_forensic_log(df, args.forensic_log)
