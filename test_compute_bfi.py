#!/usr/bin/env python3
"""
pytest unit tests for compute_bfi.py (updated)

Covers:
- Strict schema validation (missing columns, malformed binary values).
- Conditional mutual information math for H_norm with a hand-computed example.
- Edge cases (H(R) == 0 -> H_norm == 1).
- Stage reliability score behavior.
- End-to-end analyze() validity and bounds.
- Forensic logger: produces JSONL, content matches computed intermediate distributions and pipeline outputs.

Run with: pytest -q
"""

import os
import json
import math
from datetime import datetime
import pandas as pd
import numpy as np
import pytest

import compute_bfi as cb
from scipy.stats import entropy as sp_entropy

# Helper: write a CSV to tmp_path and return the path
def write_csv(tmp_path, name, header, rows):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(r + "\n")
    return str(p)

def test_load_dataset_missing_topic_raises(tmp_path):
    header = "id,label,lexical,ast,oracle,vendor_output"
    rows = [
        "r1,1,1,0,1,resp1",
        "r2,0,0,1,0,resp2",
    ]
    path = write_csv(tmp_path, "missing_topic.csv", header, rows)
    with pytest.raises(ValueError) as exc:
        cb.load_dataset(path)
    assert "missing required column" in str(exc.value).lower()

def test_load_dataset_malformed_binary_raises(tmp_path):
    header = "id,label,lexical,ast,oracle,topic,vendor_output"
    rows = [
        "r1,1,maybe,0,1,finance,resp1",
        "r2,0,0,1,0,finance,resp2",
    ]
    path = write_csv(tmp_path, "malformed_binary.csv", header, rows)
    with pytest.raises(ValueError) as excinfo:
        cb.load_dataset(path)
    msg = str(excinfo.value).lower()
    assert "column 'lexical' contains" in msg or "column 'lexical' contains non-binary" in msg

def test_load_dataset_success_and_types(tmp_path):
    header = "id,label,lexical,ast,oracle,topic,vendor_output"
    rows = [
        "r1,1,1,0,1,finance,resp1",
        "r2,0,0,1,0,finance,resp2",
        "r3,0,1,0,0,healthcare,resp3",
    ]
    path = write_csv(tmp_path, "valid.csv", header, rows)
    df = cb.load_dataset(path)
    for col in ("lexical", "ast", "oracle", "label"):
        assert col in df.columns
        assert df[col].dtype.kind in ("i", "u")
        assert set(df[col].unique()).issubset({0,1})
    assert "topic" in df.columns
    assert df["topic"].dtype == object
    assert not (df["topic"].astype(str).str.strip() == "").any()

def test_conditional_mutual_information_manual_example():
    """
    Hand-computed example to assert compute_conditional_mutual_information and normalized H_norm.
    Data and manual derivation are the same as the canonical example:
      N=4, r counts: r1:2, r2:1, r3:1 -> H(R)=1.5
      H(R|T) = 0.5, H(R|Y,T)=0.0 -> I=0.5 -> H_norm = 1 - 0.5/1.5 = 2/3
    """
    rows = [
        {"lexical":1,"ast":0,"oracle":0,"label":1,"topic":"A"},
        {"lexical":1,"ast":0,"oracle":0,"label":0,"topic":"A"},
        {"lexical":0,"ast":1,"oracle":0,"label":1,"topic":"B"},
        {"lexical":0,"ast":1,"oracle":1,"label":0,"topic":"B"},
    ]
    df = pd.DataFrame(rows)
    # Compute distributions manually (mirror compute_bfi)
    N = len(df)
    df["r"] = df.apply(lambda row: (int(row["lexical"]), int(row["ast"]), int(row["oracle"])), axis=1)
    counts_r = df.groupby("r").size().to_dict()
    p_r_vals = np.array(list(counts_r.values()), dtype=float) / float(N)
    H_R = float(sp_entropy(p_r_vals, base=2))
    assert pytest.approx(H_R, rel=1e-12) == 1.5

    H_R_given_T = 0.0
    for t, group in df.groupby("topic"):
        total_t = len(group)
        counts_r_t = group.groupby("r").size().to_dict()
        p_vals = np.array(list(counts_r_t.values()), dtype=float) / float(total_t)
        H_r_t = float(sp_entropy(p_vals, base=2)) if p_vals.size > 0 else 0.0
        H_R_given_T += (total_t / float(N)) * H_r_t
    assert pytest.approx(H_R_given_T, rel=1e-12) == 0.5

    H_R_given_Y_T = 0.0
    for (y, t), group in df.groupby(["label", "topic"]):
        total_yt = len(group)
        counts_r_yt = group.groupby("r").size().to_dict()
        p_vals = np.array(list(counts_r_yt.values()), dtype=float) / float(total_yt)
        H_r_yt = float(sp_entropy(p_vals, base=2)) if p_vals.size > 0 else 0.0
        H_R_given_Y_T += (total_yt / float(N)) * H_r_yt
    assert pytest.approx(H_R_given_Y_T, rel=1e-12) == 0.0

    I_R_Y_given_T = H_R_given_T - H_R_given_Y_T
    assert pytest.approx(I_R_Y_given_T, rel=1e-12) == 0.5
    expected_H_norm = 1.0 - (I_R_Y_given_T / H_R)
    assert pytest.approx(expected_H_norm, rel=1e-12) == 2.0/3.0

    # Compare against compute_bfi implementation
    cmi = cb.compute_conditional_mutual_information(df)
    assert cmi["N"] == N
    assert pytest.approx(cmi["H_R"], rel=1e-12) == H_R
    assert pytest.approx(cmi["H_R_given_T"], rel=1e-12) == H_R_given_T
    assert pytest.approx(cmi["H_R_given_Y_T"], rel=1e-12) == H_R_given_Y_T
    assert pytest.approx(cmi["I_R_Y_given_T"], rel=1e-12) == I_R_Y_given_T
    assert pytest.approx(cmi["H_norm"], rel=1e-12) == expected_H_norm

def test_normalized_entropy_all_R_same_returns_one():
    rows = [
        {"lexical":1,"ast":0,"oracle":0,"label":1,"topic":"A"},
        {"lexical":1,"ast":0,"oracle":0,"label":0,"topic":"A"},
        {"lexical":1,"ast":0,"oracle":0,"label":1,"topic":"B"},
    ]
    df = pd.DataFrame(rows)
    H_norm = cb.normalized_entropy(df)
    assert pytest.approx(H_norm, rel=1e-12) == 1.0

def test_stage_reliability_score_constant_behaviour():
    cfg = dict(cb.DEFAULT_CONFIG)
    cfg["bootstrap_samples"] = 50
    cfg["random_seed"] = 123
    cb.CONFIG = cfg

    rows = [
        {"lexical":1,"ast":0,"oracle":0,"label":1,"topic":"t"},
        {"lexical":1,"ast":0,"oracle":0,"label":0,"topic":"t"},
        {"lexical":1,"ast":0,"oracle":0,"label":1,"topic":"t"},
    ]
    df = pd.DataFrame(rows)
    srs = cb.stage_reliability_score(df)
    assert pytest.approx(srs["lexical"], rel=1e-12) == 1.0
    assert pytest.approx(srs["ast"], rel=1e-12) == 1.0
    assert pytest.approx(srs["oracle"], rel=1e-12) == 1.0
    assert pytest.approx(srs["mean"], rel=1e-12) == 1.0

def test_analyze_end_to_end_small_dataset(tmp_path):
    header = "id,label,lexical,ast,oracle,topic,vendor_output"
    rows = [
        "a,1,1,0,0,finance,respA",
        "b,0,0,1,0,finance,respB",
        "c,1,0,1,1,cybersecurity,respC",
        "d,0,0,0,0,education,respD",
    ]
    path = write_csv(tmp_path, "end2end.csv", header, rows)
    df = cb.load_dataset(path)
    res = cb.analyze(df)
    assert "per_stage_rates" in res
    assert "conditional_mi" in res
    cond = res["conditional_mi"]
    # Ensure conditional MI entries present and in bounds
    for k in ("H_R", "H_R_given_T", "H_R_given_Y_T", "I_R_Y_given_T", "H_norm"):
        assert k in cond
        assert isinstance(cond[k], (int, float))
    assert 0.0 <= cond["H_norm"] <= 1.0
    assert 0.0 <= res["lut_weight"] <= 1.0
    assert 0.0 <= res["hfi_normalized"] <= 1.0
    assert 0.0 <= res["bfi"] <= 100.0

def test_forensic_logger_writes_jsonl_and_content(tmp_path):
    # Prepare a small valid dataframe
    rows = [
        {"lexical":1,"ast":0,"oracle":0,"label":1,"topic":"finance"},
        {"lexical":0,"ast":1,"oracle":0,"label":0,"topic":"finance"},
        {"lexical":1,"ast":1,"oracle":1,"label":1,"topic":"cybersecurity"},
    ]
    df = pd.DataFrame(rows)
    # Ensure config small bootstrap for speed
    cb.CONFIG = dict(cb.DEFAULT_CONFIG)
    cb.CONFIG["bootstrap_samples"] = 50
    cb.CONFIG["random_seed"] = 1

    out_path = tmp_path / "forensic.jsonl"
    # Write forensic log
    cb.write_forensic_log(df, str(out_path), run_id="test-run-123", extra={"note":"unit-test"})

    # Read back and assert single JSON line present
    with open(out_path, "r", encoding="utf-8") as fh:
        lines = [l.strip() for l in fh if l.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    # Top-level sanity checks
    assert payload["run_id"] == "test-run-123"
    assert "timestamp" in payload
    # Parse timestamp format (basic check)
    datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
    assert payload["dataset"]["N"] == 3
    # Distributions in payload should match compute_conditional_mutual_information
    cmi = cb.compute_conditional_mutual_information(df)
    # Compare p_r keys and values
    assert set(payload["distributions"]["p_r"].keys()) == set(cmi["p_r"].keys())
    for k, v in payload["distributions"]["p_r"].items():
        assert pytest.approx(v, rel=1e-12) == pytest.approx(cmi["p_r"][k], rel=1e-12)
    # Check conditional MI scalars
    for key in ("H_R", "H_R_given_T", "H_R_given_Y_T", "I_R_Y_given_T", "H_norm"):
        assert pytest.approx(payload["conditional_mi"][key], rel=1e-12) == pytest.approx(cmi[key], rel=1e-12)
    # Check pipeline outputs were included
    assert "pipeline" in payload
    assert "bfi" in payload["pipeline"]
    assert 0.0 <= payload["pipeline"]["bfi"] <= 100.0
