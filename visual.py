import matplotlib.pyplot as plt
import numpy as np

def save_overlay_3signals(
    rec_id,
    snr_db,
    clean,      # clean_dec
    noisy,      # mixed_dec
    denoised,   # y_mixed_dbg (최종 출력)
    fs=360.0,
    start_sec=10.0,
    win_sec=20.0,
):
    start = int(start_sec * fs)
    end   = int((start_sec + win_sec) * fs)

    clean = np.asarray(clean)[start:end]
    noisy = np.asarray(noisy)[start:end]
    denoised = np.asarray(denoised)[start:end]

    # 겹쳐 보기 좋게 표준화(z-score)
    def z(x):
        x = x.astype(np.float64)
        s = x.std() + 1e-12
        return (x - x.mean()) / s

    clean_z = z(clean)
    noisy_z = z(noisy)
    denoised_z = z(denoised)

    t = np.arange(len(clean_z)) / fs

    plt.figure(figsize=(14, 4))
    plt.plot(t, clean_z, label="Clean (raw)")
    plt.plot(t, noisy_z, label=f"Noisy input (SNR={snr_db} dB)", alpha=0.8)
    plt.plot(t, denoised_z, label="Denoised output", alpha=0.9)

    plt.title(f"[Record {rec_id}] Overlay: clean vs noisy vs denoised")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized amplitude (z-score)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"overlay_{rec_id}_SNR{snr_db}dB.png", dpi=200)
    plt.close()