# common/config.py
"""
공통 실험 설정 파일 (Global experiment configuration)

❗중요 원칙
- 모델 종류(DAE / UNet / Proposed)에 관계없이
- "실험 조건"은 반드시 이 파일에서만 정의한다.
- 그래야 비교 실험의 공정성과 재현성이 보장된다.
"""

from pathlib import Path
from typing import List


# ============================================================
# 데이터 경로 설정
# ============================================================

# MIT-BIH Arrhythmia Database 경로
# (wfdb가 확장자 없이 record를 찾으므로 폴더 경로만 필요)
MITDB_DIR_DEFAULT: Path = Path("/home/subi/PycharmProjects/ECG/data/MITDB_data")

# Noise Stress Test Database (NSTDB) 경로
NSTDB_DIR_DEFAULT: Path = Path("/home/subi/PycharmProjects/ECG/data/noise_data")

# 결과 출력 루트 디렉토리
# 모델별 하위 폴더 (proposed / unet / dae 등)는
# 각 실행 스크립트에서 이 경로 아래로 확장해서 사용
OUTPUT_DIR_DEFAULT: Path = Path("../outputs")


# ============================================================
# 공통 실험 파라미터
# ============================================================

# MITDB / NSTDB에서 읽기 시작하는 샘플 위치
# (0이면 record의 시작부터 사용)
START_SAMPLE_DEFAULT: int = 0

# 한 record에서 사용하는 구간 길이 (초)
# 모든 모델이 동일한 시간 구간을 보게 하기 위함
DURATION_SEC_DEFAULT: int = 30

# 목표 샘플레이트 (Hz)
# 모든 신호는 이 fs 기준으로 리샘플링 후 비교
FS_DEFAULT: int = 250

# 기본 baseline wander noise 타입
# "bw": baseline wander
# "ma": muscle artifact
# "em": electrode motion
NSTDB_RECORD_DEFAULT: str = "bw"

# 입력 SNR 실험 레벨 (dB)
# 비교 실험에서는 반드시 모든 모델이 동일한 SNR 리스트 사용
SNR_LEVELS_DEFAULT: List[int] = [0, 5, 10, 15]