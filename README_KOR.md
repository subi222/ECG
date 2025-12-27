# 실시간 ECG Baseline Wander 제거 및 형태 보존 연구 (README_KOR.md)

> **연구 목표**: 심근경색 진단에 필수적인 ECG 형태(Morphology), 특히 ST-세그먼트와 QRS 복합체를 왜곡하지 않으면서 실시간으로 기선 표류(Baseline Wander)를 제거하는 프레임워크를 제안합니다.

## 1. 연구 배경 및 목적 (Problem Definition)

기선 표류(Baseline Wander, BW)는 호흡, 신체 움직임 또는 전극 접촉 문제로 인해 발생하는 심전도(ECG) 신호의 저주파 아티팩트입니다. 기존의 고역 통과 필터링(High-pass filtering) 방식은 위상 왜곡을 유발하여 **ST-세그먼트**를 인위적으로 상승시키거나 하강시키는 문제를 일으키곤 합니다.

임상 진단에서 **ST-세그먼트**는 심근 허혈 및 경색을 판독하는 핵심 지표입니다. 따라서 기선 제거 알고리즘은 다음 두 가지 상충하는 요구사항을 동시에 만족해야 합니다:
1.  **효과적인 억제**: 저주파 드리프트를 제거하여 등전위선(Isoelectric line)을 안정화함.
2.  **형태 보존 (Morphology Preservation)**: ST-세그먼트의 원래 모양과 P파, T파의 저주파 성분을 엄격하게 보존함.

본 연구는 진단적 유효성 보존 측면에서 기존 IIR/FIR 필터보다 우수한 성능을 보이는 새로운 필터링 접근 방식을 검증하는 데 목적이 있습니다.

---

## 2. 실험 설계 및 방법론 (Methodology)

알고리즘을 정량적으로 평가하기 위해 표준 PhysioNet 데이터베이스를 활용한 합성 테스트 파이프라인을 구축하였습니다.

### 2.1. 데이터셋 (Datasets)
*   **기준 신호 (Clean Reference)**: [MIT-BIH Arrhythmia Database (MITDB)](https://physionet.org/content/mitdb/)
    *   기선 제거 성능 평가를 위한 "정답(Ground Truth)" 신호로 간주합니다.
*   **노이즈 소스 (Noise Source)**: [MIT-BIH Noise Stress Test Database (NSTDB)](https://physionet.org/content/nstdb/)
    *   실제 환경에서 녹음된 기선 표류 데이터(`bw` 레코드)를 깨끗한 신호에 합성합니다.

### 2.2. 평가 파이프라인
1.  **노이즈 합성**: 깨끗한 MITDB 신호($x_{clean}$)에 다양한 신호 대 잡음비(SNR: 0dB, 5dB, 10dB, 15dB)로 NSTDB 기선 표류($n_{bw}$)를 섞습니다.
    $$ x_{noisy} = x_{clean} + \alpha \cdot n_{bw} $$
2.  **알고리즘 처리**: 제안하는 알고리즘을 $x_{noisy}$에 적용하여 추정된 깨끗한 신호 $\hat{x}_{clean}$을 얻습니다.
3.  **지표 계산**: $\hat{x}_{clean}$과 원본 $x_{clean}$을 비교하여 성능을 측정합니다.

---

## 3. 평가지표 (Evaluation Metrics)

연구의 신뢰성 확보를 위해 학술적으로 검증된 신호 품질 평가 지표를 사용합니다. 모든 수식은 `metrics.py`에 구현되어 있습니다.

### 3.1. 신호 대 잡음비 (SNR)
원본 신호의 전력 대비 잔류 노이즈 및 왜곡의 비율을 측정합니다.

$$ \text{SNR}_{dB} = 10 \log_{10} \left( \frac{\sum_{i} x_{clean}[i]^2}{\sum_{i} (x_{clean}[i] - \hat{x}_{clean}[i])^2} \right) $$

### 3.2. 평균 제곱근 오차 (RMSE)
오차 벡터의 평균적인 크기를 정량화합니다.

$$ \text{RMSE} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (x_{clean}[i] - \hat{x}_{clean}[i])^2 } $$

### 3.3. 정규화된 평균 제곱근 편차 (PRD)
압축 및 필터링 연구에서 널리 쓰이는 왜곡률 측정 지표입니다.

$$ \text{PRD} = \sqrt{ \frac{\sum_{i=1}^{N} (x_{clean}[i] - \hat{x}_{clean}[i])^2}{\sum_{i=1}^{N} (x_{clean}[i] - \bar{x}_{clean})^2} } \times 100 $$
*(단, $\bar{x}_{clean}$은 원본 신호의 평균값)*

---

## 4. 디렉토리 구조 (File Structure)

| 파일명 | 설명 |
| :--- | :--- |
| `run_synthetic_test.py` | **메인 실험 스크립트**. 본 연구의 핵심 실행 파일입니다. [MITDB + 노이즈] 합성 파이프라인을 구동하고, 성능 평가지표 산출 및 결과 보고서 생성을 총괄합니다. |
| `baseline_array.py` | **배열 기반 처리 엔진**. 실제 연구를 위해 NumPy 배열 입력을 지원하도록 리팩토링된 핵심 모듈입니다. 기존의 JSON 기반 입력을 탈피하여, 연구에 필수적인 동적 노이즈 합성 및 대량 데이터 처리가 가능하도록 최적화되었습니다. |
| `baseline.py` | **디버깅 및 연구 모듈**. 초기 알고리즘 개발 및 과정별 로직 검증을 위해 사용되었습니다. JSON 기반의 입력을 기본값으로 지원하며, 상세한 디버깅 로그와 실험 코드를 포함하고 있습니다. |
| `metrics.py` | **평가지표 라이브러리**. SNR, RMSE, PRD 등 신호 품질 평가를 위한 수학적 수식들이 구현되어 있습니다. |
| `compare_models/` | **비교 모델 연구**. 딥러닝 기반 기선 제거 방법인 "Improved DAE" (Xiong et al., 2016)의 재현 코드를 포함합니다. |

---

## 5. 딥러닝 모델 비교 분석 (Improved DAE)

제안 알고리즘의 우수성을 검증하기 위해, 대표적인 딥러닝 기반 기선 제거 기법인 **Improved DAE** (Xiong et al., 2016)를 논문에 기술된 그대로 완벽하게 재현(Reproduction)하여 비교하였습니다.

*   **비교 목적**: 전통적 신호처리 방식(제안 기법)과 딥러닝 방식(DAE) 간의 노이즈 제거 성능(SNR) 및 형태 보존력(RMSE) 비교.
*   **구현 상세**:
    *   **구조**: 101 $\to$ 50 $\to$ 50 $\to$ 101 (Fully Connected Autoencoder).
    *   **파이프라인**: 웨이블릿 변환(db6, level 8) $\to$ 윈도우 분할 $\to$ DAE $\to$ 신호 복원.
    *   **학습 전략**: 논문과 동일하게 탐욕적 계층별 사전 학습(Greedy Layer-wise Pretraining) 후 전체 미세 조정(Fine-tuning) 수행.
*   **실행 방법**:
    ```bash
    # 1. DAE 모델 학습 (사전 학습 + 미세 조정)
    python compare_models/Improved_DAE/train_DAE.py --epochs_pre 10 --epochs_fine 20

    # 2. 비교 벤치마크 실행 (제안 기법 vs DAE vs 웨이블릿)
    python compare_models/Improved_DAE/run_comparison.py --method all
    ```

---

## 6. 실행 방법 및 결과 재현 (Usage)

합성 테스트 벤치마크를 실행하려면 다음 커맨드를 사용하세요:

```bash
# 전체 합성 테스트 실행 및 성능 측정
python run_synthetic_test.py
```

**결과물:**
*   **콘솔**: 각 케이스별 입력/출력 SNR 및 RMSE 출력.
*   **`./synthetic_results/*.png`**: 파형 비교 삼중주 (기준 신호 vs 노이즈 신호 vs 처리 결과).
*   **`./synthetic_results/synthetic_test_results.csv`**: Mean $\pm$ Std 통계치가 포함된 정량적 데이터 테이블.

---

## 7. 향후 연구 로드맵 (Future Work)

*   **임상 데이터 검증**: 실제 ST-segment 변화가 포함된 주석 데이터셋(예: European ST-T Database)에서 성능 검증.
*   **실시간 임베디드 구현**: `baseline_array.py`를 저전력 홀터 모니터 배포를 위해 C/C++ 환경에 최적화.
*   **범용 노이즈 대응**: 근전도(EMG) 및 전극 움직임 아티팩트(EM)를 포함한 다양한 노이즈 환경으로 평가 확장.
