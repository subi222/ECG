
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.linalg import solveh_banded
from scipy.ndimage import uniform_filter1d, percentile_filter, gaussian_filter1d, maximum_filter1d
from typing import List, Tuple, Optional
import concurrent.futures

# Optional: neurokit2 for R-peak detection if r_idx is not provided.
try:
    import neurokit2 as nk
except ImportError:
    nk = None

# =============================================================================
# V37 Standalone Implementation: ECG Baseline & Morphology Preservation
# =============================================================================

def fast_percentile_filter(x, percentile, size, fs=250.0, target_fs=10.0):
    """
    메모리 효율적인 백분위수 필터 (Percentile Filter).
    대규모 데이터 처리를 위해 내부적으로 다운샘플링을 수행한 후 보간(Interpolation)하여
    연산 속도를 비약적으로 향상시키면서도 기저선의 거시적 흐름을 유지합니다.
    
    Args:
        x: 입력 신호
        percentile: 필터링할 백분위 (예: 15는 하단 기저선 추적용)
        size: 필터 윈도우 크기 (샘플 수)
        fs: 입력 신호 샘플링 레이트
        target_fs: 다운샘플링 타켓 레이트 (기본 10Hz)
    """
    n_in = x.size
    win_size = min(n_in, max(3, int(size)))
    if n_in < 100:
        return percentile_filter(x, percentile=percentile, size=win_size, mode='nearest')
    
    q = int(max(1, fs // target_fs))
    if q <= 1:
        return percentile_filter(x, percentile=percentile, size=win_size, mode='nearest')
    
    n_ds_max = n_in // q
    if n_ds_max < 5:
        return percentile_filter(x, percentile=percentile, size=win_size, mode='nearest')
        
    x_ds = x[:n_ds_max * q].reshape(-1, q).mean(axis=1)
    size_ds = min(x_ds.size, max(3, win_size // q))
    size_ds = max(3, size_ds)
    b_ds = percentile_filter(x_ds, percentile=percentile, size=size_ds, mode='nearest')
    
    t_ds = np.linspace(0, n_in, b_ds.size)
    t_full = np.arange(n_in)
    return np.interp(t_full, t_ds, b_ds)

def fast_moving_stats(x, win):
    """
    이동 평균 및 이동 표준편차를 계산합니다.
    scipy.ndimage.uniform_filter1d를 사용하여 O(N) 시간 복잡도로 빠르게 통계량을 산출합니다.
    신호의 국소적인 변동성(Local Visibility/Volume)을 측정하는 데 사용됩니다.
    """
    n = x.size
    w = min(n, max(1, int(win)))
    m = uniform_filter1d(x, size=w, mode='nearest')
    m2 = uniform_filter1d(x*x, size=w, mode='nearest')
    s = np.sqrt(np.maximum(m2 - m*m, 0.0))
    return m, s

class BeatTemplateMemory:
    """
    ECG 박동 템플릿 메모리 관리 클래스.
    이전 박동들의 형태학적 특징을 학습(Exponential Moving Average)하여 저장하고,
    새로 들어온 박동이 기존 패턴과 얼마나 다른지(Distance) 계산하여 아티팩트를 감지합니다.
    """
    def __init__(self, p_len: int = 80, st_len: int = 120, alpha: float = 0.18):
        self.p_len = int(p_len)
        self.st_len = int(st_len)
        self.alpha = float(alpha)
        self.p_tpl = None

    @staticmethod
    def resample_fixed(seg: np.ndarray, length: int) -> np.ndarray:
        """박동 세그먼트를 고정된 길이로 리샘플링하여 비교 가능하게 만듭니다."""
        if seg is None or seg.size <= 1:
            return np.zeros(length, float)
        f = interp1d(np.linspace(0, 1, seg.size), seg.astype(float), kind='linear', fill_value='extrapolate', assume_sorted=True)
        return f(np.linspace(0, 1, length))

    def update(self, p_seg: Optional[np.ndarray], st_seg: Optional[np.ndarray] = None) -> float:
        """
        템플릿을 업데이트하고 현재 세그먼트와 템플릿 간의 거리(1 - corr)를 반환합니다.
        가중치 alpha를 사용하여 최신 박동 패턴을 점진적으로 반영합니다.
        """
        dist = 0.0
        if p_seg is not None and p_seg.size > 3:
            rp = self.resample_fixed(p_seg, self.p_len)
            rp -= np.mean(rp)
            norm = np.sqrt(np.mean(rp**2)) + 1e-9
            rp /= norm
            
            if self.p_tpl is None:
                self.p_tpl = rp
            else:
                tpl_norm = self.p_tpl / (np.sqrt(np.mean(self.p_tpl**2)) + 1e-9)
                corr = np.vdot(rp, tpl_norm) / self.p_len
                dist = np.clip(1.0 - corr, 0.0, 1.0)
                self.p_tpl = (1.0 - self.alpha) * self.p_tpl + self.alpha * rp
        return dist

def baseline_asls_masked(y, lam=1e6, p=0.008, niter=10, mask=None, decim=1, use32=True):
    """
    적응형 비대칭 최소제곱법 (Adaptive AsLS).
    기저선 추정의 핵심 엔진으로, 2차 차분 페널티를 사용하여 평활한 기저선을 구합니다.
    
    V37 특이점:
    - mask: True인 구간은 기저선 피팅을 수행하고, False인 구간은 무시(보호)합니다.
    - lam: 국소 강성 계수(Lambda)가 벡터로 전달되어 아티팩트 구간에서만 선택적으로 유연해질 수 있습니다.
    - decim: 연산 가속을 위한 데시메이션 지원.
    """
    dt = np.float32 if use32 else float
    x = np.asarray(y, dt)
    N = len(y)
    if N < 3: return np.zeros(N, dt)
    
    if decim > 1:
        n = (N // decim) * decim
        if n > 0:
            lam_ds = lam[:n].reshape(-1, decim).mean(1) if isinstance(lam, np.ndarray) else lam
            p_ds = p[:n].reshape(-1, decim).mean(1) if isinstance(p, np.ndarray) else p
            zds = baseline_asls_masked(x[:n].reshape(-1, decim).mean(1), lam_ds, p_ds, niter, None, 1, use32)
            z = np.repeat(zds, decim)
            return np.append(z, np.full(N-z.size, z[-1], dt))[:N]
    
    g = np.ones(N, dt) if mask is None else np.where(mask, 1.0, 1e-3).astype(dt)
    #l_vec = np.asarray(lam, dt) if isinstance(lam, np.ndarray) else np.full(N - 2, float(lam), dt)
    #if l_vec.size < N - 2:
        #l_vec = np.append(l_vec, np.full(N - 2 - l_vec.size, l_vec[-1], dt))

    # --- FIX: lam 벡터 길이를 N-2로 정규화 ---
    if isinstance(lam, np.ndarray):
        l_vec = np.asarray(lam, dt).ravel()

        # lam이 길이 N(샘플별)로 들어오는 경우가 많아서 N-2로 맞춰야 함
        if l_vec.size == N:
            l_vec = l_vec[1:-1]  # 중앙 N-2 사용(가장 자연스러움)
        elif l_vec.size > N - 2:
            l_vec = l_vec[:N - 2]  # 너무 길면 자르기
        elif l_vec.size < N - 2:
            l_vec = np.append(l_vec, np.full(N - 2 - l_vec.size, l_vec[-1], dt))
    else:
        l_vec = np.full(N - 2, float(lam), dt)
    # --- FIX END ---

    ab = np.zeros((3, N), dt)
    ab[0, 2:] = l_vec
    ab[1, 1] = -2.0 * l_vec[0]
    if N > 3: ab[1, 2:-1] = -2.0 * (l_vec[:-1] + l_vec[1:])
    ab[1, -1] = -2.0 * l_vec[-1]
    ab[2, 0] = l_vec[0] + 1e-6
    ab[2, 1] = 4.0 * l_vec[0] + l_vec[1]
    if N > 4: ab[2, 2:-2] = l_vec[:-2] + 4.0 * l_vec[1:-1] + l_vec[2:]
    ab[2, -2] = l_vec[-2] + 4.0 * l_vec[-1]
    ab[2, -1] = l_vec[-1]
    
    bd = ab[2].copy()
    w = np.ones(N, dt)
    z = np.zeros(N, dt)
    
    for _ in range(niter):
        wg = w * g
        ab[2, :] = bd + wg

        eps = np.float64(1e-8)  # 또는 dt(1e-6~1e-8 사이 추천)

        for _ in range(niter):
            wg = w * g
            ab[2, :] = bd + wg

            # --- SPD 보장용 diagonal jitter (핵심 수정 위치) ---
            ab[2, :] += eps
            # ---------------------------------------------------

            z = solveh_banded(ab, wg * x, lower=False, check_finite=False)
            w = p * (x > z) + (1.0 - p) * (x <= z)
        return z

        z = solveh_banded(ab, wg * x, lower=False, check_finite=False)
        w = p * (x > z) + (1.0 - p) * (x <= z)
    return z

def _find_breaks(y, fs, jump_k=15.0, r_idx=None):
    """
    신호의 단절 및 급격한 기저선 도약(Artifact Jump)을 감지합니다.
    국소 Slew-rate(변화율)를 분석하여 정상적인 QRS 박동과 구별되는 비정상적인 도약을 찾아냅니다.
    """
    y = np.asarray(y); N = y.size
    if N < 2: return []
    grad = np.abs(np.diff(y, prepend=y[0]))
    mad_global = 1.4826 * (np.median(np.abs(grad - np.median(grad))) + 1e-12)
    jumps = np.flatnonzero(grad > jump_k * mad_global)
    
    avg_slew = uniform_filter1d(grad, int(2.0 * fs)|1)
    struct_jumps = np.flatnonzero(grad > 30.0 * (avg_slew + 1e-6))
    jumps = np.unique(np.concatenate((jumps, struct_jumps)))
    
    breaks = []
    if jumps.size:
        if r_idx is not None and len(r_idx) > 0:
            rp = int(0.12 * fs)
            near_idx = np.searchsorted(r_idx, jumps)
            dist_l = np.where(near_idx > 0, jumps - r_idx[np.clip(near_idx - 1, 0, len(r_idx)-1)], np.inf)
            dist_r = np.where(near_idx < len(r_idx), r_idx[np.clip(near_idx, 0, len(r_idx)-1)] - jumps, np.inf)
            jumps = jumps[(dist_l >= rp) & (dist_r >= rp)]

        if jumps.size:
            gs = int(0.4 * fs)
            last_b = -gs
            for j in jumps:
                if j - last_b > gs:
                    breaks.append(int(j)); last_b = j
    return sorted(list(set(breaks)))

def _dilate_mask(mask, fs, pad_s=0.3):
    """불리언 마스크 구역을 좌우로 확장(Dilation)하여 보호 구역을 넓힙니다."""
    p = int(round(pad_s * fs))
    if p <= 0 or not mask.any(): return mask
    flat = mask.ravel()
    diff = np.diff(flat.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    starts_ex = np.maximum(0, starts - p)
    ends_ex = np.minimum(mask.size, ends + p)
    out = np.zeros_like(mask, dtype=bool)
    for s, e in zip(starts_ex, ends_ex):
        out[s:e] = True
    return out

def _baseline_tp_spline(y, fs, r_idx=None):
    """
    P-Chip Spline을 사용하여 등전위 구역(PQ Segment)을 잇는 기저선을 생성합니다.
    ECG 신호에서 가장 안정적인 구역인 PQ junction을 앵커 포인트로 활용하여
    박동 간의 미세한 기저선 변동을 보정합니다.
    """
    x = np.asarray(y, float); N = len(x)
    if r_idx is None or len(r_idx) < 2: return np.zeros(N)
    
    knot_x = [0]; knot_y = [float(np.median(x[:int(0.1*fs)]) if N > 0 else 0)]
    is_irregular = np.std(np.diff(r_idx)) / (np.mean(np.diff(r_idx)) + 1e-9) > 0.22
    win_scale = 2.5 if is_irregular else 1.0
    pq_start_offset = int(0.06 * fs * win_scale)
    pq_end_offset = int(0.02 * fs)
    
    for r in r_idx:
        start, end = int(r - pq_start_offset), int(r - pq_end_offset)
        if start > 0 and end < N:
            ax = (start + end) // 2
            ay = float(np.median(x[start:end+1]))
            knot_x.append(ax); knot_y.append(ay)

    knot_x.append(N-1); knot_y.append(knot_y[-1])
    knot_x, knot_y = np.asarray(knot_x), np.asarray(knot_y)
    unique_idx = np.concatenate(([True], np.diff(knot_x) > 0))
    knot_x, knot_y = knot_x[unique_idx], knot_y[unique_idx]
    if len(knot_x) < 2: return np.zeros(N)
    return PchipInterpolator(knot_x, knot_y)(np.arange(N))

def rr_isoelectric_clamp(y, fs, r_idx=None, t0_ms=80, t1_ms=300):
    """
    박동 간 등전위 구역(Isoelectric area)의 중전위값을 계산하여 기저선을 0으로 고정합니다.
    박동 사이에 발생하는 미세한 부유 현상을 완벽하게 억제하는 테더(Tether) 역할을 합니다.
    """
    x = np.asarray(y, float); N = len(y)
    if r_idx is None or len(r_idx) < 2: return np.zeros(N)
    t0, t1 = int(t0_ms*1e-3*fs), int(t1_ms*1e-3*fs)
    r_trim = r_idx[:-1]
    starts, ends = np.maximum(0, r_trim + t0), np.minimum(N, r_trim + t1)
    min_len = max(5, int(0.04*fs))
    valid = (ends - starts) >= min_len
    if not np.any(valid): return np.zeros(N)
    v_st, px = starts[valid], (starts[valid] + ends[valid]) // 2
    L = t1 - t0
    if L > 1:
        in_bounds = (v_st >= 0) & (v_st + L <= N)
        if not np.any(in_bounds): return np.zeros(N)
        idx2d = v_st[in_bounds][:, None] + np.arange(L)
        py = np.median(x[idx2d], axis=1)
        px = px[in_bounds]
    else:
        py = x[px]
    br = np.interp(np.arange(N), px, py)
    return br - np.median(br)

def _apply_pre_beat_ripple_suppression(y, fs, r_idx, noise_mad):
    """
    R-peak 직전(P-wave 구역)에 발생하는 미세한 고주파 리플을 억제합니다.
    노이즈가 심한 경우 P-wave와 노이즈가 섞여 판독이 어려운 것을 방지하기 위해
    해당 구역만 선택적으로 가우시안 평활화를 적용합니다.
    """
    if noise_mad < 10.0: return y
    y_out = y.copy(); N = len(y)
    sigma_px = int(0.02 * fs) | 1
    starts, ends = np.maximum(0, r_idx - int(0.22*fs)), np.maximum(0, r_idx - int(0.06*fs))
    for s, e in zip(starts, ends):
        if e <= s or s >= N: continue
        seg_smooth = gaussian_filter1d(y[s:e], sigma_px)
        alpha = np.clip((noise_mad - 10.0) / 40.0, 0.0, 0.8)
        y_out[s:e] = y[s:e] * (1.0 - alpha) + seg_smooth * alpha
    return y_out

def _apply_beat_synchronized_polish(y, fs, r_idx, noise_mad):
    """
    V37의 핵심 기능인 비트 동기화 폴리싱(Beat-Synchronized Polish).
    신호 자체에서 추출한 '중앙값 박동 템플릿'을 원본 신호와 적응적으로 합성합니다.
    임상적인 파형의 형태(Morphology)를 유지하면서도 불규칙한 미세 노이즈를 효과적으로 제거합니다.
    """
    N = len(y)
    pre_s, post_s = int(0.25 * fs), int(0.40 * fs)
    beats = [y[r-pre_s:r+post_s] for r in r_idx if r-pre_s>=0 and r+post_s<=N]
    if not beats: return y
    template = np.median(beats, axis=0)
    ratio = np.clip((noise_mad - 5.0) / 100.0, 0.0, 0.4)
    if ratio < 0.05: return y
    y_out = y.copy()
    for r in r_idx:
        s, e = r - pre_s, r + post_s
        if s < 0 or e > N: continue
        y_out[s:e] = y_out[s:e] * (1.0 - ratio) + template * ratio
    return y_out

def _apply_morphology_guided_denoise(sig, fs, is_baseline_area, noise_mad):
    """
    형태학 인지 기반의 적응형 고주파 노이즈 제거.
    QRS 복합체 구간은 선명도를 위해 약하게 필터링하고, 기저선 구간(Isoelectric)은
    노이즈 제거를 위해 강력하게 필터링(Low-pass)하는 가변 필터링 전략을 수행합니다.
    """
    if noise_mad < 5.0 or len(sig) < 10: return sig
    fc = 35.0 if noise_mad > 80.0 else 45.0
    sos_lp = signal.butter(4, fc, 'low', fs=fs, output='sos')
    y_lp = signal.sosfiltfilt(sos_lp, sig)
    sos_qrs = signal.butter(4, 120.0, 'low', fs=fs, output='sos')
    y_qrs = signal.sosfiltfilt(sos_qrs, sig)
    w_baseline = uniform_filter1d(is_baseline_area.astype(float), int(0.04 * fs)|1)
    return y_qrs * (1.0 - w_baseline) + y_lp * w_baseline

def v37_baseline_correction(y, fs, asls_lam=5e7, asls_p=0.5, r_idx=None, qrs_aware=True, adaptive_denoise=True):
    """
    V37 Hybrid BL++ 기저선 교정 알고리즘 통합 함수.
    
    동작 순서:
    1. Polarity Detection: 신호의 반전 상태를 확인하고 정방향으로 보정한 뒤 연산합니다.
    2. Adaptive Stiffness Setting: 전역 노이즈 레벨을 측정하여 AsLS의 기본 강성과 마스크 확장 폭을 결정합니다.
    3. Non-linear Constraint Calculation: 15차 승수 수식을 사용하여 아티팩트 발생 지점에서 강성을 0에 가깝게 떨어뜨립니다.
    4. Hierarchical Optimization: Spline 기반의 앵커링과 AsLS 최적화를 병합하여 하이브리드 기저선을 도출합니다.
    5. Morphological Policing: 비트 동기화 템플릿 합성과 적응형 고주파 필터링으로 최종 신호를 정화합니다.
    """
    x_raw = np.asarray(y, float); N = x_raw.size
    if N < 8: return x_raw, np.zeros_like(x_raw)
    
    med_val = np.nanmedian(x_raw)
    x0_centered = x_raw - med_val
    sample = x0_centered[::max(1, N // 5000)]
    is_inverted = (np.mean((sample - np.mean(sample))**3) / (np.std(sample)**3 + 1e-9)) < -0.4 
    x = -x0_centered if is_inverted else x0_centered
    
    # Early 1st pass
    w0 = int(round(2.5 * fs)) | 1
    b0 = fast_percentile_filter(x, 15, w0, fs, 15.0)
    
    p_grad = np.abs(np.diff(x))
    global_noise_mad = np.nanmedian(p_grad) / 0.6745
    mask_expand = np.clip(1.0 + (global_noise_mad - 20.0) / 40.0, 1.0, 2.0)
    asls_lam_b = max(asls_lam, 10.0) * np.clip(1.0 + ((global_noise_mad - 20.0) / 40.0)**2, 1.0, 100.0)

    # QRS-Aware Masking
    if r_idx is None and qrs_aware and nk:
        try: r_idx = np.array(nk.ecg_peaks(x, fs)[1].get("ECG_R_Peaks", []), int)
        except: pass
    
    base_mask = np.ones(N, bool)
    if r_idx is not None and len(r_idx) > 0:
        rr_ints = np.diff(r_idx)
        avg_rr = np.median(rr_ints) if rr_ints.size > 0 else fs
        rr_s = (np.concatenate(([avg_rr], rr_ints)) + np.concatenate((rr_ints, [avg_rr]))) / (2.0 * fs)
        pre_r = (0.18 * mask_expand * fs).astype(int)
        k_baz = np.clip(0.48 * mask_expand, 0.1, 0.7)
        post_r = (k_baz * np.sqrt(rr_s) * fs).astype(int)
        change = np.zeros(N + 1, int)
        np.add.at(change, np.maximum(0, r_idx - pre_r), 1)
        np.add.at(change, np.minimum(N, r_idx + post_r + 1), -1)
        base_mask = np.cumsum(change)[:-1] == 0
    
    # Adaptive Stiffness
    grad = np.abs(np.diff(x, prepend=x[0]))
    mad_ws = int(2.0 * fs) | 1
    local_mad = uniform_filter1d(np.abs(grad - uniform_filter1d(grad, mad_ws)), mad_ws) * 1.4826 + 1e-12
    local_mad_clp = np.clip(local_mad, 0.5 * global_noise_mad, 3.0 * global_noise_mad)
    z_grad = np.where(grad / (2.0 * local_mad_clp) < 2.5, 0.0, grad / (2.0 * local_mad_clp))
    z_comb = maximum_filter1d(z_grad, size=int(0.40 * fs))
    
    # V22 Artifact punch-through
    z_abs = np.abs(x - uniform_filter1d(x, int(3.0 * fs)|1)) / (2.0 * local_mad_clp + 1e-9)
    z_comb = np.maximum(z_comb, z_abs)
    base_mask |= (z_abs > 7.5)
    
    lam_local = asls_lam_b / (1.0 + np.power(z_comb, 15.0))
    # Stiffness Smoothing
    lam_smooth = np.minimum(np.exp(uniform_filter1d(np.log(asls_lam_b / (1.0+2.0*(local_mad/global_noise_mad)) + 1e-12), int(1.0*fs))), lam_local)
    
    # Stage 1: Anchor Baseline
    b_spline = _baseline_tp_spline(x - b0, fs, r_idx=r_idx)
    x_res = x - b0 - b_spline
    
    # Stage 2: Adaptive ASLS Fine-tuning
    b1 = baseline_asls_masked(x_res, lam_smooth, 0.5 if asls_p==0.5 else asls_p, 10, base_mask if qrs_aware else None, 1, False)
    
    b_f = b0 + b_spline + b1
    
    # Final Tether
    iso = rr_isoelectric_clamp(x - b_f, fs, r_idx=r_idx)
    b_f += (iso - np.median(iso))
    b_f = uniform_filter1d(b_f, size=int(0.1*fs)|1)

    y_corr = (-(x - b_f) if is_inverted else (x - b_f))
    baseline = x_raw - y_corr
    
    if r_idx is not None and len(r_idx) > 0:
        y_corr = _apply_pre_beat_ripple_suppression(y_corr, fs, r_idx, global_noise_mad)
        if len(r_idx) > 5 and global_noise_mad > 5.0:
            y_corr = _apply_beat_synchronized_polish(y_corr, fs, r_idx, global_noise_mad)
            
    if adaptive_denoise:
        y_corr = _apply_morphology_guided_denoise(y_corr, fs, base_mask, global_noise_mad)
        baseline = x_raw - y_corr

    return y_corr, baseline

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1. ECG 데이터 및 노이즈 생성 (Synthetic ECG Generation)
    # =========================================================================
    # 아래 섹션은 알고리즘 검증을 위한 노이즈 섞인 ECG 데이터를 생성하는 구간입니다.

    fs = 250.0
    duration = 10.0 # seconds
    t = np.arange(int(fs * duration)) / fs
    
    # Synthetic QRS (Simple triangles)
    clean_ecg = np.zeros_like(t)
    r_peaks = np.arange(0.5, duration, 0.8) * fs
    r_peaks = r_peaks.astype(int)
    for r in r_peaks:
        clean_ecg[r-5:r+5] = 1.0 - np.abs(np.linspace(-1, 1, 10))
    
    # Baseline Wander (Slow Sinusoids)
    bw = 0.5 * np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.sin(2 * np.pi * 0.05 * t)
    
    # Artifact (Sharp Jump)
    artifact = np.zeros_like(t)
    artifact[int(fs*5):] = 1.5
    
    # High Frequency Noise
    noise = 0.05 * np.random.randn(len(t))
    
    noisy_ecg = clean_ecg + bw + artifact + noise
    # ========================== 여기까지 노이즈 생성 ==========================

    
    # 2. Apply V37 Baseline Correction
    print("Processing ECG with V37 Standalone...")
    corrected_ecg, baseline_est = v37_baseline_correction(noisy_ecg, fs, r_idx=r_peaks)
    
    # 3. Plot Results
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t, noisy_ecg, label="Noisy Input", color='gray', alpha=0.5)
    plt.plot(t, clean_ecg, label="Ground Truth (Clean)", color='blue', linestyle='--')
    plt.title("ECG with Baseline Wander & Artifact")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(t, noisy_ecg, label="Noisy Input", color='gray', alpha=0.3)
    plt.plot(t, baseline_est, label="Estimated Baseline (V37)", color='red', linewidth=2)
    plt.title("Baseline Extraction")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(t, clean_ecg, label="Ground Truth", color='blue', alpha=0.5)
    plt.plot(t, corrected_ecg, label="Corrected ECG (V37)", color='darkorange')
    plt.title("Final Corrected Signal")
    plt.legend()

    plt.tight_layout()
    plt.savefig("v37_demo_result.png", dpi=150)
    # plt.show()  ← 제거하거나 주석 처리
    print("Done.")
