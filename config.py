"""
统一路径与超参数配置
用法：
    from config import DATA_DIR, OUTPUT_DIR, CORR_PATH, DATASETS
"""
from __future__ import annotations
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR: Path = Path(
    os.environ.get("DATA_DIR", str(PROJECT_ROOT / "data"))
).expanduser().resolve()

OUTPUT_DIR: Path = Path(
    os.environ.get("OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))
).expanduser().resolve()

CORR_PATH: Path = DATA_DIR / "corr_dim_scaled.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS: dict[str, Path] = {
    "sz50":      DATA_DIR / "sz50_index_data.csv",
    "hs300":     DATA_DIR / "hs300_index_data.csv",
    "cnpc":      DATA_DIR / "cnpc_data.csv",
    "cmb":       DATA_DIR / "cmb_data.csv",
    "maotai":    DATA_DIR / "maotai_data.csv",
    "yili":      DATA_DIR / "yili_data.csv",
    "pingan":    DATA_DIR / "pingan_data.csv",
    "gree":      DATA_DIR / "gree_data.csv",
    "ningde":    DATA_DIR / "ningde_data.csv",
    "dongfang":  DATA_DIR / "dongfang_data.csv",
    "btc":       DATA_DIR / "BTC_data.csv",
}

DATASET_LABELS: dict[str, str] = {
    "sz50": "上证50指数", "hs300": "沪深300指数",
    "cnpc": "中石油", "cmb": "招商银行", "maotai": "贵州茅台",
    "yili": "伊利股份", "pingan": "中国平安", "gree": "格力电器",
    "ningde": "宁德时代", "dongfang": "东方财富", "btc": "比特币",
}

# Stage1 DS-LDE 超参数默认值
DEFAULT_TAU: int = 1
DEFAULT_EMBED_DIM: int = 23
DEFAULT_N_STEPS: int = 4
DEFAULT_SEED: int = 42

def get_data_path(name: str) -> Path:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS.keys())}")
    path = DATASETS[name]
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}\nSet DATA_DIR in .env")
    return path

if __name__ == "__main__":
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"DATA_DIR     = {DATA_DIR}  (exists={DATA_DIR.exists()})")
    print(f"OUTPUT_DIR   = {OUTPUT_DIR}")
    print("\nDatasets:")
    for k, v in DATASETS.items():
        print(f"  [{'OK' if v.exists() else 'MISSING':7s}] {k:12s} -> {v.name}")
