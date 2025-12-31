# common/dataset_split.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import re
import numpy as np


def list_mitdb_records(mitdb_dir: Path) -> List[int]:
    """
    MIT-BIH Arrhythmia Database(MITDB) 폴더에서
    사용 가능한 record ID를 자동으로 수집한다.

    - *.hea 파일을 기준으로 record를 판단
      (wfdb는 확장자 없이 record 이름을 사용)
    - 예: '100.hea' → record id = 100

    Parameters
    ----------
    mitdb_dir : Path
        MITDB_data 폴더 경로

    Returns
    -------
    records : List[int]
        사용 가능한 record id 목록 (정렬된 상태)
    """
    recs: List[int] = []

    # MITDB 폴더 내의 모든 header 파일 탐색
    for p in mitdb_dir.glob("*.hea"):
        # record 이름은 숫자로만 이루어져야 함 (예: 100.hea)
        m = re.match(r"^(\d+)\.hea$", p.name)
        if m:
            recs.append(int(m.group(1)))

    # 중복 제거 + 정렬
    return sorted(set(recs))


def split_records(
    records: List[int],
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    MITDB record를 record 단위로 train / validation으로 분리한다.

    ❗중요 설계 원칙
    - 반드시 "record 단위"로 분리한다.
      (같은 record의 window가 train/val에 섞이면 데이터 leakage 발생)
    - seed를 고정해 실험 재현성을 확보한다.

    Parameters
    ----------
    records : List[int]
        전체 record id 목록
    val_ratio : float, default=0.15
        validation set 비율 (예: 0.15 → 15%)
    seed : int, default=42
        랜덤 시드 (비교 실험 시 모든 모델에서 동일해야 함)

    Returns
    -------
    train_records : List[int]
        train에 사용할 record id 목록
    val_records : List[int]
        validation에 사용할 record id 목록
    """
    # NumPy RandomState 사용 → seed 고정으로 항상 동일한 분리 보장
    rng = np.random.RandomState(seed)

    # record id는 정렬된 상태에서 permutation
    records = sorted(records)
    perm = rng.permutation(records)

    # validation record 개수 계산 (최소 1개는 확보)
    n_val = max(1, int(round(len(records) * val_ratio)))

    # 앞부분을 validation, 나머지를 train으로 사용
    val_recs = sorted(perm[:n_val].tolist())
    train_recs = sorted(perm[n_val:].tolist())

    return train_recs, val_recs


# common/dataset_split.py

# MITDB record 목록을 가져와
# train / val / test를 한 번에 나누고
# common/splits.json으로 저장하는 역할만 함

import json
from pathlib import Path
from typing import Dict, List

def create_splits_json(
    mitdb_dir,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    out_path: Path = None,
) -> Dict[str, List[int]]:
    """
    Create a fixed record-wise train/val/test split and save it as JSON.

    - Split is done at the RECORD level (not window level) to avoid data leakage.
    - Ratios must sum to 1.0.
    - Intended to be shared across all models for fair comparison.
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train/val/test ratios must sum to 1.0"

    # --- 핵심 수정 ---
    mitdb_dir = Path(mitdb_dir)

    # Import here to avoid circular import
    from .dataset_split import list_mitdb_records, split_records

    # 1) Get all MITDB record IDs
    records = list_mitdb_records(mitdb_dir)
    records = sorted(records)

    # 2) First split: (train + val) vs test
    train_val_records, test_records = split_records(
        records,
        val_ratio=test_ratio,
        seed=seed
    )

    # 3) Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_records, val_records = split_records(
        train_val_records,
        val_ratio=val_ratio_adjusted,
        seed=seed
    )

    splits = {
        "train": sorted(train_records),
        "val": sorted(val_records),
        "test": sorted(test_records),
    }

    # 4) Save to JSON
    if out_path is None:
        out_path = Path(__file__).resolve().parent / "splits.json"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print("[Split Created]")
    print(f"  Train: {len(splits['train'])} records")
    print(f"  Val  : {len(splits['val'])} records")
    print(f"  Test : {len(splits['test'])} records")
    print(f"  Saved to: {out_path}")

    return splits