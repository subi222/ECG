# common/io_wfdb.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import wfdb


def load_mitdb_wfdb(
    mitdb_dir: Path,
    record: int,
    start_sample: int,
    duration_sec: int,
    prefer_channels: Tuple[str, ...] = ("MLII", "V5"),
) -> Tuple[np.ndarray, int]:
    """
    MITDB에서 ECG 1채널을 읽어 지정 구간만 반환한다.

    Parameters
    ----------
    mitdb_dir : Path
        MITDB_data 폴더 경로 (예: .../data/MITDB_data)
    record : int
        레코드 번호 (예: 100)
    start_sample : int
        시작 샘플 인덱스 (fs 기준)
    duration_sec : int
        읽을 길이(초)
    prefer_channels : tuple[str,...]
        채널 선택 우선순위. 기본은 ("MLII","V5").
        해당 채널이 없으면 0번 채널 사용.

    Returns
    -------
    ecg_segment : np.ndarray
        shape=(N,) 1D ECG 구간 (float64)
    fs : int
        샘플레이트(Hz)
    """
    rec_path = mitdb_dir / str(record)  # wfdb는 확장자 없이 읽음

    # wfdb로 전체 샘플 읽기 (sig: (N, n_ch))
    sig, fields = wfdb.rdsamp(str(rec_path))
    fs = int(fields["fs"])
    names: List[str] = list(fields.get("sig_name", []))

    # 채널 선택: prefer_channels 순서대로 찾고 없으면 0번 채널
    ch = 0
    for cname in prefer_channels:
        if cname in names:
            ch = names.index(cname)
            break

    ecg = sig[:, ch].astype(np.float64)

    # 구간 자르기 (start/end 범위 보호)
    start = max(0, int(start_sample))
    end = start + int(fs * duration_sec)
    end = min(end, len(ecg))

    # start가 end를 넘어가면(이상 입력) 빈 배열이 아니라 안전하게 0길이 반환
    if start >= end:
        return np.asarray([], dtype=np.float64), fs

    return ecg[start:end], fs


def load_nstdb_noise(
    nstdb_dir: Path,
    record: str,
    start_sample: int,
    duration_sec: int,
    fs: int,
) -> Tuple[np.ndarray, int]:
    """
    NSTDB(노이즈 DB)에서 noise 1채널을 읽어 지정 구간만 반환한다.

    - wfdb는 확장자 없이 읽으므로 rec_path는 ".../bw" 같은 형태
    - 요청 구간이 신호 길이를 초과하면 wrap padding으로 길이를 맞춤

    Parameters
    ----------
    nstdb_dir : Path
        noise_data 폴더 경로
    record : str
        노이즈 레코드 이름 (예: "bw", "ma", "em")
    start_sample : int
        시작 샘플 인덱스 (fs 기준)
    duration_sec : int
        읽을 길이(초)
    fs : int
        목표 샘플레이트(Hz). (이미 FS_TARGET로 리샘플된 noise를 넣는 설계라면 동일해야 함)

    Returns
    -------
    noise_segment : np.ndarray
        shape=(N,) 1D noise 구간 (float64)
    fs : int
        샘플레이트(Hz) (일반적으로 입력 fs 그대로 반환)
    """
    rec_path = nstdb_dir / record

    # 헤더 존재 확인(경로 실수/파일 누락을 빨리 잡기 위함)
    if not (nstdb_dir / f"{record}.hea").exists():
        raise FileNotFoundError(f"NSTDB header not found: {rec_path}.hea")

    sig, fields = wfdb.rdsamp(str(rec_path))
    fs_read = int(fields.get("fs", fs))  # 방어적으로 fs를 fields에서 읽되, 없으면 인자로 받은 fs 사용

    # NSTDB는 보통 2채널일 수 있으나, 여기서는 0번 채널만 사용
    noise = sig[:, 0].astype(np.float64)

    start = max(0, int(start_sample))
    end = start + int(fs * duration_sec)

    # 요청 구간이 길면 wrap 패딩으로 확장 (반복되는 노이즈를 이어 붙이는 효과)
    if end > len(noise):
        pad_len = end - len(noise)
        noise = np.pad(noise, (0, pad_len), mode="wrap")

    if start >= end:
        return np.asarray([], dtype=np.float64), fs_read

    return noise[start:end], fs_read
