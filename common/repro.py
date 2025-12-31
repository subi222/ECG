# common/repro.py
"""
Reproducibility utilities (실험 재현성 보장)

❗중요 원칙
- 이 함수는 성능을 향상시키기 위한 것이 아니다.
- 실험을 다시 실행했을 때 "같은 결과"를 얻기 위한 장치다.
- 비교 실험에서는 모든 모델이 동일한 seed를 사용해야 한다.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Python / NumPy / PyTorch의 랜덤 시드를 고정한다.

    이 함수를 호출하면:
    - 데이터 split 결과
    - DataLoader shuffle 순서
    - 모델 가중치 초기화
    등이 가능한 한 동일하게 재현된다.

    Parameters
    ----------
    seed : int, default=42
        랜덤 시드 값 (비교 실험 시 전 모델 공통 사용 권장)
    """
    # Python 기본 random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (GPU, 사용 중일 경우)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CUDA 연산을 가능한 한 결정적으로(deterministic) 만들기
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False