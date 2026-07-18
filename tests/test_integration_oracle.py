import pytest, requests
import compute_bfi as cb
import pandas as pd

def test_pipeline_write_forensic_log():
    rows = [{"lexical":1,"ast":0,"oracle":0,"label":1,"topic":"cybersecurity"}]
    df = pd.DataFrame(rows)
    cb.write_forensic_log(df, "test.jsonl", run_id="test-1")
    with open("test.jsonl", "r") as f:
        assert "run_id" in f.readline()
