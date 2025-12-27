# Real-time ECG Baseline Wander Removal & Morphology Preservation Research

> **Project Goal**: To propose a real-time framework that removes baseline wander while maintaining the diagnostic integrity of ECG morphology—specifically the ST-segment and QRS complex—which is critical for accurate diagnosis of myocardial infarction.

## 1. Problem Definition

Baseline wander (BW) is a low-frequency artifact in Electrocardiogram (ECG) signals caused by respiration, body movement, or electrode contact issues. Traditional high-pass filtering often introduces phase distortion, leading to artificially elevated or depressed **ST-segments**.

In clinical diagnostics, the **ST-segment** is the key indicator for myocardial ischemia and infarction. Therefore, a baseline removal algorithm must satisfy two conflicting requirements:
1.  **Effective Suppression**: Remove low-frequency drift to stabilize the isoelectric line.
2.  **Morphology Preservation**: Strictly preserve the original shape of the ST-segment and low-frequency components of the P and T waves.

This research aims to validate a novel filtering approach that outperforms standard IIR/FIR filters in terms of diagnostic feature preservation.

---

## 2. Methodology & Experimental Design

To quantitatively evaluate the algorithm, we devised a synthetic testing pipeline using standard PhysioNet databases.

### 2.1. Datasets
*   **Ground Truth (Clean)**: [MIT-BIH Arrhythmia Database (MITDB)](https://physionet.org/content/mitdb/)
    *   Regarded as "clean" reference signals for the purpose of BW removal evaluation.
*   **Noise Source**: [MIT-BIH Noise Stress Test Database (NSTDB)](https://physionet.org/content/nstdb/)
    *   Real-world baseline wander recordings (e.g., `bw` record) injected into clean signals.

### 2.2. Evaluation Pipeline
1.  **Synthetic Injection**: Mix clean MITDB signals ($x_{clean}$) with NSTDB baseline wander ($n_{bw}$) at varying Signal-to-Noise Ratios (SNR: 0dB, 5dB, 10dB, 15dB).
    $$ x_{noisy} = x_{clean} + \alpha \cdot n_{bw} $$
2.  **Processing**: Apply the proposed algorithm to $x_{noisy}$ to obtain $\hat{x}_{clean}$.
3.  **Metric Calculation**: Compare $\hat{x}_{clean}$ against the original $x_{clean}$.

---

## 3. Evaluation Metrics

We strictly follow accepted academic standards for signal quality assessment. Codes are implemented in `metrics.py`.

### 3.1. Signal-to-Noise Ratio (SNR)
Measures the power ratio between the clean signal and the residual noise/distortion.

$$ \text{SNR}_{dB} = 10 \log_{10} \left( \frac{\sum_{i} x_{clean}[i]^2}{\sum_{i} (x_{clean}[i] - \hat{x}_{clean}[i])^2} \right) $$

### 3.2. Root Mean Square Error (RMSE)
Quantifies the average magnitude of the error vector.

$$ \text{RMSE} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (x_{clean}[i] - \hat{x}_{clean}[i])^2 } $$

### 3.3. Percent Root Mean Square Difference (PRD)
A normalized measure of distortion, widely used in compression and filtering literature.

$$ \text{PRD} = \sqrt{ \frac{\sum_{i=1}^{N} (x_{clean}[i] - \hat{x}_{clean}[i])^2}{\sum_{i=1}^{N} (x_{clean}[i] - \bar{x}_{clean})^2} } \times 100 $$
*(Where $\bar{x}_{clean}$ is the mean of the clean signal)*

---

## 4. File Structure

| File | Description |
| :--- | :--- |
| `run_synthetic_test.py` | **Main Experiment Script**. The primary entry point for benchmarking. It orchestrates the [MITDB + Noise] mixing pipeline, invokes the processing engine, and generates statistical reports. |
| `baseline_array.py` | **Main Processing Engine (Array-based)**. Refactored to support direct NumPy array inputs. This transition from legacy JSON-based processing was necessary to support the dynamic noise-mixing required for this research. |
| `baseline.py` | **Research & Debugging Module**. Used for initial algorithm development and step-by-step logic verification. It contains legacy support for JSON inputs and extensive debugging helpers. |
| `metrics.py` | **Metric Library**. Mathematical implementations of signal quality indicators: SNR, RMSE, PRD, and Cosine Similarity. |
| `compare_models/` | **Comparison Benchmarks**. Contains reproduction code for deep learning models, specifically the "Improved DAE" (Xiong et al., 2016). |

---

## 5. Comparative Analysis (Improved DAE)

This project includes a high-fidelity reproduction of **Improved DAE** (Xiong et al., 2016), a deep learning-based baseline removal method, to demonstrate the effectiveness of our proposed algorithm.

*   **Objective**: Compare noise suppression (SNR) and morphology preservation (RMSE) between our signal processing approach and a standard Denoising Autoencoder.
*   **Implementation Details**:
    *   **Structure**: 101 $\to$ 50 $\to$ 50 $\to$ 101 (Fully Connected).
    *   **Pipeline**: Wavelet Transform (db6, level 8) $\to$ Windowing $\to$ DAE $\to$ Reconstruction.
    *   **Training**: Greedy Layer-wise Pretraining + Fine-tuning (Reproduced as per paper).
*   **Execution**:
    ```bash
    # 1. Train the DAE (Pretraining + Fine-tuning)
    python compare_models/Improved_DAE/train_DAE.py --epochs_pre 10 --epochs_fine 20

    # 2. Run Comparison Benchmark
    python compare_models/Improved_DAE/run_comparison.py --method all
    ```

---

## 6. Usage & Reproducibility

To reproduce the synthetic evaluation results:

```bash
# Run the full synthetic test benchmark
python run_synthetic_test.py
```

**Output:**
*   **Console**: Prints Input/Output SNR and RMSE for each case.
*   **`./synthetic_results/*.png`**: Visualization triplets (Clean Reference vs. Noisy Input vs. Processed Output).
*   **`./synthetic_results/synthetic_test_results.csv`**: Detailed quantitative table including Mean $\pm$ Std statistics.

---

## 7. Future Work / Research Roadmap

*   **Clinical Validation**: Validate algorithm performance on manually annotated ST-elevation datasets (e.g., European ST-T Database).
*   **Real-time Embedded Porting**: Optimize `baseline_array.py` for C/C++ deployment on low-power Holter monitors.
*   **Generic Noise Robustness**: Extend evaluation to include muscle artifacts (EMG) and electrode motion artifacts (EM).
