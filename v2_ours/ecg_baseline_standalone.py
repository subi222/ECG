# ecg_baseline_standalone.py
# Standalone module for ECG Baseline Correction (Hybrid + Adaptive Stiffness)
# Auto-generated. Contains all necessary dependencies for baseline_hybrid_plus_adaptive.

import numpy as np
import hashlib
import json
import concurrent.futures
from collections import defaultdict
from time import perf_counter
from threading import Lock
from typing import Optional, List, Tuple, Set
from pathlib import Path
from scipy import signal
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.linalg import solveh_banded
from scipy.ndimage import uniform_filter1d, percentile_filter, binary_dilation
from scipy.signal import savgol_filter, decimate

try:
    import neurokit2 as nk
except ImportError:
    nk = None

# ==============================================================================
# 1. Utilities & Profiling
# ==============================================================================

_PROF = defaultdict(lambda: {"calls": 0, "total": 0.0})
_PROF_LOCK = Lock()

def _prof_add(name: str, dt: float):
    with _PROF_LOCK:
        d = _PROF[name]
        d["calls"] += 1
        d["total"] += float(dt)

def profiled(name: Optional[str] = None):
    def deco(fn):
        label = name or fn.__name__
        def wrapped(*args, **kwargs):
            t0 = perf_counter()
            try: return fn(*args, **kwargs)
            finally: _prof_add(label, perf_counter() - t0)
        wrapped.__name__ = fn.__name__
        wrapped.__doc__ = fn.__doc__
        return wrapped
    return deco

def profiler_report(topn: int = 30):
    rows = []
    for k, v in _PROF.items():
        calls = v["calls"] or 1; total = v["total"]; avg = total / calls
        rows.append((k, calls, total, avg))
    rows.sort(key=lambda r: r[2], reverse=True)
    if not rows: return []
    headers = ("function", "calls", "total_ms", "avg_ms"); formatted = []
    for name, calls, total, avg in rows[:topn]:
        formatted.append((name, f"{calls:,}", f"{total * 1000:,.2f}", f"{avg * 1000:,.2f}"))
    widths = [len(h) for h in headers]
    for row in formatted:
        for idx, cell in enumerate(row): widths[idx] = max(widths[idx], len(cell))
    def _fmt_row(row_cells):
        pieces = []
        for idx, cell in enumerate(row_cells):
            width = widths[idx]
            pieces.append(cell.ljust(width) if idx == 0 else cell.rjust(width))
        return "| " + " | ".join(pieces) + " |"
    def _border(char="-"): return "+" + "+".join(char * (w + 2) for w in widths) + "+"
    print(f"\n[Profiler]"); print(_border("-")); print(_fmt_row(headers)); print(_border("=")); [print(_fmt_row(row)) for row in formatted]; print(_border("-"))
    return rows

def _fingerprint(arr: np.ndarray) -> tuple:
    if arr is None or arr.size == 0: return (0, 0, 0.0, 0.0)
    sample_size = min(arr.size, 1024)
    h = hashlib.blake2b(arr[:sample_size].tobytes(), digest_size=8).hexdigest()
    return (arr.size, hash(arr.dtype.str), float(arr[0]), float(arr[-1]), h)

def cached_numpy(maxsize=32):
    from functools import wraps
    cache = {}
    lock = Lock()
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            key_parts = []
            for a in args:
                if isinstance(a, np.ndarray): key_parts.append(_fingerprint(a))
                else: key_parts.append(a)
            for k, v in sorted(kwargs.items()):
                if isinstance(v, np.ndarray): key_parts.append((k, _fingerprint(v)))
                else: key_parts.append((k, v))
            key = tuple(key_parts)
            with lock:
                if key in cache: return cache[key]
            res = fn(*args, **kwargs)
            with lock:
                if len(cache) >= maxsize:
                    del cache[next(iter(cache))]
                cache[key] = res
            return res
        return wrapped
    return deco

def chunked_processing(func, x, fs, chunk_size=500000, overlap=50000, **kwargs):
    x = np.asarray(x)
    N = x.size
    if N <= chunk_size: return func(x, fs, **kwargs)
    step = chunk_size - overlap
    num_chunks = int(np.ceil((N - overlap) / step))
    
    # Run first chunk to detect return signature
    res0 = func(x[:chunk_size], fs, **kwargs)
    is_tuple = isinstance(res0, tuple)
    
    if is_tuple:
        out_bufs = [np.zeros(N, dtype=res0[0].dtype) for _ in res0]
        w_buf = np.zeros(N, dtype=float)
        def _accum(res, start, end, w):
            for i, arr in enumerate(res): out_bufs[i][start:end] += arr * w
    else:
        out_buf = np.zeros(N, dtype=res0.dtype)
        w_buf = np.zeros(N, dtype=float)
        def _accum(res, start, end, w): out_buf[start:end] += res * w

    for i in range(num_chunks):
        start = i * step
        end = min(start + chunk_size, N)
        actual_len = end - start
        chunk = x[start:end]
        res = func(chunk, fs, **kwargs)
        w = np.ones(actual_len, float)
        if i > 0: fl = min(overlap, actual_len); w[:fl] = np.linspace(0, 1, fl)
        if i < num_chunks - 1: fl = min(overlap, actual_len); w[-fl:] = np.linspace(1, 0, fl)
        _accum(res, start, end, w)
        w_buf[start:end] += w
        
    mask = w_buf > 1e-9
    if is_tuple:
        final_res = []
        for buf in out_bufs:
            buf[mask] /= w_buf[mask]
            final_res.append(buf)
        return tuple(final_res)
    else:
        out_buf[mask] /= w_buf[mask]
        return out_buf


# ==============================================================================
# 2. Helper Functions (Signal Processing)
# ==============================================================================

def fast_percentile_filter(x, percentile, size, fs=250.0, target_fs=10.0):
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
    b_ds = percentile_filter(x_ds, percentile=percentile, size=size_ds, mode='nearest')
    t_ds = np.linspace(0, n_in, b_ds.size)
    t_full = np.arange(n_in)
    f = interp1d(t_ds, b_ds, kind='linear', fill_value='extrapolate', assume_sorted=True)
    return f(t_full)

def fast_moving_stats(x, win):
    n = x.size
    w = min(n, max(1, int(win)))
    m = uniform_filter1d(x, size=w, mode='nearest')
    m2 = uniform_filter1d(x*x, size=w, mode='nearest')
    s = np.sqrt(np.maximum(m2 - m*m, 0.0))
    return m, s

@profiled()
def _dilate_mask(mask, fs, pad_s=0.45):
    p = int(round(pad_s*fs))
    if p <= 0 or not mask.any(): return mask
    flat = mask.ravel()
    diff = np.diff(flat.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    if starts.size == 0: return mask
    starts_ex = np.maximum(0, starts - p)
    ends_ex = np.minimum(mask.size, ends + p)
    out = np.zeros_like(mask, dtype=bool)
    if starts_ex.size > 0:
        for s, e in zip(starts_ex, ends_ex):
            out[s:e] = True
    return out

@profiled()
def make_qrs_mask(y, fs=250, rp_ms=150, ts_ms=80, te_ms=350, r_idx=None):
    N = len(y)
    if r_idx is None:
        if not nk: return np.zeros(N, bool)
        try: r_idx = np.array(nk.ecg_peaks(y, fs)[1].get("ECG_R_Peaks", []), int)
        except: return np.zeros(N, bool)
    if not r_idx.size: return np.zeros(N, bool)
    
    valid_r = r_idx[(r_idx >= 0) & (r_idx < N)]
    rp = int(rp_ms * 1e-3 * fs)
    te = int(te_ms * 1e-3 * fs)
    starts = np.maximum(0, valid_r - rp)
    ends = np.minimum(N, valid_r + te)
    
    change = np.zeros(N+1, int)
    change[starts] += 1
    change[ends] -= 1
    prot = np.cumsum(change)[:-1].astype(bool)
    return (~prot)

@profiled()
def _find_breaks(y, fs, min_dur_s=1.0, jump_k=12.0, r_idx=None):
    y = np.asarray(y)
    N = y.size
    if N < 2: return []
    dy = np.abs(np.diff(y, prepend=y[0]))
    is_flat = dy < 1e-9
    v_max, v_min = np.max(y), np.min(y)
    is_sat = (y == v_max) | (y == v_min)
    is_bad = is_flat | is_sat
    breaks = []
    min_samples = int(min_dur_s * fs)
    bad_idx = np.flatnonzero(is_bad)
    if bad_idx.size >= min_samples:
        diff_bad = np.diff(bad_idx, prepend=bad_idx[0]-1)
        starts = bad_idx[diff_bad > 1]
        ends = bad_idx[np.concatenate((diff_bad[1:] > 1, [True]))]
        for s, e in zip(starts, ends):
            if (e - s) >= min_samples: breaks.append(int((s + e) // 2))

    grad = np.abs(np.diff(y, prepend=y[0]))
    m_grad_global = np.median(grad)
    mad_global = 1.4826 * (np.median(np.abs(grad - m_grad_global)) + 1e-12)
    jumps = np.flatnonzero(grad > jump_k * mad_global)
    avg_slew = uniform_filter1d(grad, int(2.0 * fs)|1)
    struct_jumps = np.flatnonzero(grad > 30.0 * (avg_slew + 1e-6))
    jumps = np.unique(np.concatenate((jumps, struct_jumps)))
    
    if jumps.size and r_idx is not None and len(r_idx) > 0:
        rp = int(0.12 * fs)
        r_idx = np.sort(r_idx)
        near_r = np.searchsorted(r_idx, jumps)
        mask_near = np.zeros_like(jumps, bool)
        for k in range(jumps.size):
            idx_r = near_r[k]
            if idx_r < len(r_idx) and abs(jumps[k] - r_idx[idx_r]) < rp:
                mask_near[k] = True; continue
            if idx_r > 0 and abs(jumps[k] - r_idx[idx_r-1]) < rp:
                mask_near[k] = True
        jumps = jumps[~mask_near]
    if jumps.size:
        gs = int(0.4 * fs)
        last_b = -gs
        for j in jumps:
            if j - last_b > gs:
                breaks.append(int(j)); last_b = j
    return sorted(list(set(breaks)))


# ==============================================================================
# 3. Core Logic (Baseline Correction)
# ==============================================================================

@cached_numpy(maxsize=128)
@profiled()
def baseline_asls_masked(y, lam=1e6, p=0.008, niter=10, mask=None, decim=1, use32=True, ws=None, tlen=1500):
    dt = np.float32 if use32 else float
    x = np.asarray(y, dt)
    N = len(y)
    if N<3: return np.zeros(N, dt)
    q = decim
    if q<=1 and N>tlen: q = int(np.ceil(N/tlen))
    if q>1:
        n = (N//q)*q
        if n<q: return np.zeros(N, dt)
        lam_ds = lam
        if isinstance(lam, np.ndarray):
            lam_ds = lam[:n].reshape(-1, q).mean(1)
        zds = baseline_asls_masked(x[:n].reshape(-1, q).mean(1), lam_ds, p, niter, None, 1, use32, ws, tlen)
        z = np.repeat(zds, q)
        return np.append(z, np.full(N-z.size, z[-1], dt))[:N]
    
    g = np.ones(N, dt) if mask is None else np.where(mask, 1.0, 1e-3).astype(dt)
    if isinstance(lam, np.ndarray):
        l_vec = np.asarray(lam, dt)
        if l_vec.size < N - 2: l_vec = np.append(l_vec, np.full(N - 2 - l_vec.size, l_vec[-1], dt))
        elif l_vec.size > N - 2: l_vec = l_vec[:N - 2]
    else:
        l_vec = np.full(N - 2, float(lam), dt)

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
    w, z = np.ones(N, dt), np.zeros(N, dt)
    ap = None
    for _ in range(niter):
        wg = w * g
        ab[2, :] = bd + wg
        z = solveh_banded(ab, wg * x, lower=False)
        active = (x > z)
        if ap is not None and np.array_equal(active, ap): break
        ap = active
        w[active] = p
        w[~active] = 1 - p
    return z

@cached_numpy(maxsize=16)
@profiled()
def _baseline_tp_spline(y, fs, r_idx=None):
    x = np.asarray(y, float)
    N = len(x)
    if r_idx is None or len(r_idx) < 2: return np.zeros(N)
    knot_x = [0]
    knot_y = [float(np.median(x[:int(0.1*fs)]) if N > 0 else 0)]
    pq_start_offset = int(0.06 * fs)
    pq_end_offset = int(0.02 * fs)
    for r in r_idx:
        r = int(r)
        start = r - pq_start_offset
        end = r - pq_end_offset
        if start > 0 and end < N:
            ax = (start + end) // 2
            ay = float(np.median(x[start:end+1]))
            knot_x.append(ax)
            knot_y.append(ay)
    knot_x.append(N-1)
    knot_y.append(knot_y[-1])
    knot_x = np.asarray(knot_x)
    knot_y = np.asarray(knot_y)
    unique_idx = np.concatenate(([True], np.diff(knot_x) > 0))
    knot_x = knot_x[unique_idx]
    knot_y = knot_y[unique_idx]
    if len(knot_x) < 2: return np.zeros(N)
    interpolator = PchipInterpolator(knot_x, knot_y)
    return interpolator(np.arange(N))

@profiled()
def rr_isoelectric_clamp(y, fs, r_idx=None, t0_ms=80, t1_ms=300):
    x = np.asarray(y, float); N = len(y)
    if r_idx is None or len(r_idx) < 2:
        if nk: 
            try: r_idx = np.array(nk.ecg_peaks(x, fs)[1].get("ECG_R_Peaks", []), int)
            except: pass
    if r_idx is None or len(r_idx) < 2: return np.zeros(N)
    
    t0, t1 = int(t0_ms*1e-3*fs), int(t1_ms*1e-3*fs)
    r_trim = r_idx[:-1]
    starts = np.maximum(0, r_trim + t0)
    ends = np.minimum(N, r_trim + t1)
    min_len = max(5, int(0.04*fs))
    valid = (ends - starts) >= min_len
    if not np.any(valid): return np.zeros(N)
    v_st, v_en = starts[valid], ends[valid]
    px = (v_st + v_en) // 2
    py = np.array([np.median(x[s:e]) for s, e in zip(v_st, v_en)])
    br = np.interp(np.arange(N), px, py)
    return br - np.median(br)

def _baseline_core(y, fs, per_win_s=2.5, per_q=15, asls_lam=5e7, asls_p=0.5, asls_decim=8, qrs_aware=True, verylow_fc=0.55, clamp_win_s=6.0, vol_win_s=0.6, vol_gain=2.0, lam_floor_ratio=0.003, hard_cut=True, break_pad_s=0.30, rr_cap_enable=True, rr_eps_up=5.0, rr_eps_dn=8.0, rr_t0_ms=80, rr_t1_ms=320, r_idx=None, qrs_mask=None, lam_bins=3, min_seg_s=10.0, max_seg_s=60.0):
    x = np.asarray(y, float)
    N = x.size
    if N < 8: return np.zeros_like(x), np.zeros_like(x)
    w0 = int(round(per_win_s * fs)); w0 += (1 - w0 % 2)
    x0 = x - np.median(x[np.isfinite(x)])
    b0 = fast_percentile_filter(x0, int(per_q), w0, fs, 15.0)
    
    if qrs_mask is not None:
        base_mask = qrs_mask.astype(bool)
    elif qrs_aware:
        if r_idx is None and nk:
            try: r_idx = np.array(nk.ecg_peaks(x, fs)[1].get("ECG_R_Peaks", []), int)
            except: r_idx = np.array([], int)
        base_mask = np.ones(N, bool)
        if r_idx is not None and len(r_idx) > 0:
            rr_intervals = np.diff(r_idx)
            if rr_intervals.size > 0:
                avg_rr = np.median(rr_intervals)
                for i, r in enumerate(r_idx):
                    rr_prev = rr_intervals[i-1] if i > 0 else avg_rr
                    rr_next = rr_intervals[i] if i < len(rr_intervals) else avg_rr
                    rr_s = ((rr_prev + rr_next) / 2.0) / fs
                    pre_r = int(round(0.18 * fs))
                    post_r = int(round(0.48 * np.sqrt(rr_s) * fs))
                    post_r = min(post_r, int(0.7 * rr_next)) if i < len(rr_intervals) else post_r
                    base_mask[max(0, r - pre_r):min(N, r + post_r + 1)] = False
            else:
                base_mask[max(0, r_idx[0]-int(0.18*fs)):min(N, r_idx[0]+int(0.40*fs))] = False
    else:
        base_mask = np.ones(N, bool)
        
    brks = _find_breaks(x, fs, r_idx=r_idx, jump_k=20.0)
    if brks:
        prot = np.zeros(N, bool)
        prot[np.asarray(brks, int)] = True
        jml = _dilate_mask(prot, fs, break_pad_s)
        base_mask[jml] = True 
        
    grad = np.abs(np.diff(x, prepend=x[0]))
    mad_ws = int(2.0 * fs) | 1
    local_median = uniform_filter1d(grad, mad_ws)
    local_mad = uniform_filter1d(np.abs(grad - local_median), mad_ws) * 1.4826 + 1e-12
    z_grad = grad / (2.0 * local_mad)
    
    m_v, rs = fast_moving_stats(x, int(round(vol_win_s*fs))|1)
    q_vol = np.quantile(rs, 0.90) + 1e-9
    z_vol = np.clip(rs / q_vol, 0, 10)

    lam_local = asls_lam / (1.0 + np.power(z_grad, 8.0))
    lam_local = np.maximum(lam_local, 1.0)
    lam_vol_limit = asls_lam / (1.0 + float(vol_gain) * z_vol)
    lam_local = np.minimum(lam_local, lam_vol_limit)
    
    L_vol = np.log(lam_vol_limit + 1e-12)
    lam_smooth_bg = np.exp(uniform_filter1d(L_vol, int(1.0 * fs)))
    lam_smooth = np.minimum(lam_smooth_bg, lam_local)
    
    if brks:
        tw = int(0.12 * fs)
        for b in brks:
             lam_smooth[max(0,b-tw):min(N,b+tw+1)] = 1.0 
    
    cuts = [0] + [int(b) for b in (brks if hard_cut else [])] + [N]
    segs_f = []
    mxl = int(round(float(max_seg_s)*fs))
    for s0, e0 in zip(cuts[:-1], cuts[1:]):
        if e0 <= s0: continue
        for a in range(s0, e0, mxl):
            b = min(e0, a + mxl)
            if b-a > 5: segs_f.append((a, b))
            
    b1 = np.zeros(N)
    b_spline = _baseline_tp_spline(x0 - b0, fs, r_idx=r_idx)
    x_res = x0 - b0 - b_spline

    def _worker(args):
        s, e, seg, lam_vec = args
        try:
            res = baseline_asls_masked(seg, lam_vec, asls_p, 10, (None if not qrs_aware else base_mask[s:e]), asls_decim, use32=True)
            return s, e, res
        except:
             res = baseline_asls_masked(seg.astype(float), lam_vec.astype(float), asls_p, 10, (None if not qrs_aware else base_mask[s:e]), asls_decim, use32=False)
             return s, e, res

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_worker, [(s, e, x_res[s:e], lam_smooth[s:e]) for s, e in segs_f if (e-s) >= max(5, int(0.2*fs))]))
        for s, e, res in results:
            b1[s:e] = res

    b_f = b0 + b_spline + b1
    
    if rr_cap_enable:
        iso = rr_isoelectric_clamp(x - b_f, fs, r_idx=r_idx, t0_ms=rr_t0_ms, t1_ms=rr_t1_ms)
        iso -= np.median(iso)
        b_f = b_f + iso
        b_f = uniform_filter1d(b_f, size=int(0.1*fs)|1, mode='nearest')

    return x - b_f, b_f

@cached_numpy(maxsize=8)
@profiled()
def baseline_hybrid_plus_adaptive(y, fs, per_win_s=2.5, per_q=15, asls_lam=5e7, asls_p=0.5, asls_decim=8, qrs_aware=True, verylow_fc=0.55, clamp_win_s=6.0, vol_win_s=0.6, vol_gain=2.0, lam_floor_ratio=0.003, hard_cut=True, break_pad_s=0.30, rr_cap_enable=True, rr_eps_up=5.0, rr_eps_dn=8.0, rr_t0_ms=80, rr_t1_ms=320, r_idx=None, qrs_mask=None, lam_bins=3, min_seg_s=10.0, max_seg_s=60.0):
    """
    Main entry point for baseline correction.
    Uses chunking and parallel processing for optimal performance.
    """
    x = np.asarray(y, float)
    if x.size > 2_000_000:
        return chunked_processing(_baseline_core, x, fs, chunk_size=1_000_000, overlap=100_000,
                                  per_win_s=per_win_s, per_q=per_q, asls_lam=asls_lam, asls_p=asls_p,
                                  asls_decim=asls_decim, qrs_aware=qrs_aware, verylow_fc=verylow_fc,
                                  clamp_win_s=clamp_win_s, vol_win_s=vol_win_s, vol_gain=vol_gain,
                                  lam_floor_ratio=lam_floor_ratio, hard_cut=hard_cut, break_pad_s=break_pad_s,
                                  rr_cap_enable=rr_cap_enable, rr_eps_up=rr_eps_up, rr_eps_dn=rr_eps_dn,
                                  rr_t0_ms=rr_t0_ms, rr_t1_ms=rr_t1_ms, r_idx=r_idx, qrs_mask=qrs_mask,
                                  lam_bins=lam_bins, min_seg_s=min_seg_s, max_seg_s=max_seg_s)
    
    return _baseline_core(x, fs, per_win_s, per_q, asls_lam, asls_p, asls_decim, qrs_aware, verylow_fc,
                          clamp_win_s, vol_win_s, vol_gain, lam_floor_ratio, hard_cut, break_pad_s,
                          rr_cap_enable, rr_eps_up, rr_eps_dn, rr_t0_ms, rr_t1_ms, r_idx, qrs_mask,
                          lam_bins, min_seg_s, max_seg_s)


# ==============================================================================
# 4. Optional Masking Utilities
# ==============================================================================

@cached_numpy(maxsize=16)
@profiled()
def suppress_negative_sag(y, fs, win_sec=1.0, q_floor=20, k_neg=3.5, min_dur_s=0.25, pad_s=0.25, protect_qrs=True, r_idx=None, qrs_mask=None, use_fast_filter=True):
    y = np.asarray(y, float)
    N = y.size
    w = int(round(win_sec*fs))|1
    ml, pn = int(round(min_dur_s*fs)), int(round(pad_s*fs))
    if use_fast_filter:
        mv, sv = fast_moving_stats(y, w)
        fl = mv - abs(0.01*(50-q_floor))*0.1*sv
        md = mv
    else:
        fl = fast_percentile_filter(y, q_floor, w, fs, 15.0)
        md = fast_percentile_filter(y, 50, w, fs, 15.0)
    neg = np.minimum(y - md, 0.0)
    med_n = np.median(neg)
    mad_n = np.median(np.abs(neg-med_n)) + 1e-12
    mask = ((neg - med_n)/(1.4826*mad_n) < -abs(k_neg)) & (y < fl)
    if protect_qrs:
        pro = qrs_mask.astype(bool) if qrs_mask is not None else np.zeros(N, bool)
        if qrs_mask is None and nk:
            try:
                temp_r = r_idx if r_idx is not None else np.array(nk.ecg_peaks(y, fs)[1].get("ECG_R_Peaks", []), int)
                pd = int(0.12*fs)
                for r in temp_r:
                    pro[max(0,r-pd):min(N,r+pd+1)] = True
            except: pass
        mask &= (~pro)
    if not mask.any(): return mask
    if ml > 1:
        df = np.diff(mask.astype(int), prepend=0, append=0)
        st, en = np.flatnonzero(df==1), np.flatnonzero(df==-1)
        valid = (en - st) >= ml
        if not valid.any(): return np.zeros(N, bool)
        change = np.zeros(N+1, int)
        change[st[valid]] = 1
        change[en[valid]] = -1
        mask = np.cumsum(change)[:-1].astype(bool)
    if pn > 0: mask = _dilate_mask(mask, fs, pad_s)
    return mask


@cached_numpy(maxsize=16)
@profiled()
def fix_downward_steps_mask(y, fs, pre_s=0.5, post_s=0.5, gap_s=0.08, amp_sigma=5.0, amp_abs=None, min_hold_s=0.45, refractory_s=0.80, protect_qrs=True, r_idx=None, qrs_mask=None, smooth_ms=120, hop_ms=10):
    y = np.asarray(y, float)
    N = y.size
    sw = min(N, int(round(smooth_ms*1e-3*fs))|1)
    ys = uniform_filter1d(y, size=sw, mode='nearest') if smooth_ms > 0 else y
    m_ys = np.median(ys)
    mad_ys = 1.4826 * (np.median(np.abs(ys - m_ys)) + 1e-12)
    thr = max(amp_sigma * mad_ys, float(amp_abs or 0))
    S = np.concatenate(([0.0], np.cumsum(ys, dtype=float)))
    pr, po, ga, ho, re = int(round(pre_s*fs)), int(round(post_s*fs)), int(round(gap_s*fs)), int(round(min_hold_s*fs)), int(round(refractory_s*fs))
    hp = max(1, int(round(hop_ms*1e-3*fs)))
    imx = N-(ga+po+ho)-1
    if imx<=pr: return np.zeros(N, bool)
    cs = np.arange(pr, imx+1, hp)
    m1 = (S[cs]-S[cs-pr])/pr
    cp = cs+ga
    m2 = (S[cp+po]-S[cp])/po
    mh = (S[cp+ho]-S[cp])/ho
    dp = m1-m2
    cand = (dp > thr) & ((m1-mh) >= 0.6*dp)
    if protect_qrs:
        pro = qrs_mask.astype(bool) if qrs_mask is not None else np.zeros(N, bool)
        if qrs_mask is None and nk:
            try:
                temp_r = r_idx if r_idx is not None else np.array(nk.ecg_peaks(ys, fs)[1].get("ECG_R_Peaks", []), int)
                pp = int(0.12*fs)
                for r in temp_r:
                    pro[max(0,r-pp):min(N,r+pp+1)] = True
            except: pass
        if pro.any(): cand &= (~pro[cs])
    out = np.zeros(N, bool)
    if not cand.any(): return out
    cand_indices = np.where(cand)[0]
    idxs = np.argsort(-dp[cand])
    le = -1e12
    for j in idxs:
        idx_in_cs = cand_indices[j]
        s = cp[idx_in_cs]
        if s - le < re: continue
        out[s:s+ho] = True
        le = s + ho
    return out

@cached_numpy(maxsize=16)
@profiled()
def smooth_corners_mask(y, fs, L_ms=140, k_sigma=5.5, protect_qrs=True, r_idx=None, qrs_mask=None, smooth_ms=20, use_float32=True):
    y = np.asarray(y, np.float32 if use_float32 else float)
    N = y.size
    sw = min(N, int(round(smooth_ms*1e-3*fs))|1)
    ys = uniform_filter1d(y, size=sw, mode='nearest') if smooth_ms>0 else y
    d2 = np.gradient(np.gradient(ys))
    cand = np.abs((d2-np.median(d2))/(1.4826*np.median(np.abs(d2-np.median(d2)))+1e-12)) > float(k_sigma)
    if protect_qrs:
        pro = qrs_mask.astype(bool) if qrs_mask is not None else np.zeros(N, bool)
        if qrs_mask is None and nk:
            try:
                temp_r = r_idx if r_idx is not None else np.array(nk.ecg_peaks(ys, fs)[1].get("ECG_R_Peaks", []), int)
                pp = int(0.12*fs)
                for r in temp_r:
                    pro[max(0,r-pp):min(N,r+pp+1)] = True
            except: pass
        cand &= (~pro)
    if not cand.any(): return np.zeros(N, bool)
    L = int(round(L_ms*1e-3*fs))|1
    return _dilate_mask(cand, fs, L/fs)

@cached_numpy(maxsize=16)
@profiled()
def burst_mask(y, fs, win_ms=140, k_diff=7.5, k_std=3.5, pad_ms=80, protect_qrs=True, r_idx=None, qrs_mask=None, pre_smooth_ms=0, use_float32=True):
    x = np.asarray(y, np.float32 if use_float32 else float)
    N = x.size
    if pre_smooth_ms>0:
        sw = min(N, int(round(pre_smooth_ms*1e-3*fs))|1)
        x = uniform_filter1d(x, size=sw, mode='nearest')
    dy = np.gradient(x)
    mv, rv = fast_moving_stats(x, int(round(win_ms*1e-3*fs))|1)
    m_dy = np.median(dy)
    mad_dy = 1.4826 * np.median(np.abs(dy - m_dy)) + 1e-12
    m_rv = np.median(rv)
    mad_rv = 1.4826 * np.median(np.abs(rv - m_rv)) + 1e-12
    cand = (np.abs((dy - m_dy)/mad_dy) > float(k_diff)) & ((rv - m_rv)/mad_rv > float(k_std))
    if protect_qrs:
        pro = qrs_mask.astype(bool) if qrs_mask is not None else np.zeros(N, bool)
        if qrs_mask is None and nk:
            try:
                temp_r = r_idx if r_idx is not None else np.array(nk.ecg_peaks(x, fs)[1].get("ECG_R_Peaks", []), int)
                pp = int(0.12*fs)
                for r in temp_r:
                    pro[max(0,r-pp):min(N,r+pp+1)] = True
            except: pass
        cand &= (~pro)
    if cand.any() and pad_ms > 0:
        return _dilate_mask(cand, fs, pad_ms / 1000.0)
    return cand

@cached_numpy(maxsize=16)
@profiled()
def high_variance_mask(y, win=2000, k_sigma=5.0, pad=125, hop_ms=32):
    x = np.asarray(y, np.float32)
    fs = 250 
    n = x.size
    if n==0: return np.zeros(0, bool), {}
    hp = max(1, int(round(hop_ms*1e-3*fs)))
    cs = np.arange(0, n, hp)
    st, en = np.clip(cs-win//2, 0, n-1), np.clip(cs+win//2+1, 0, n)
    S1, S2 = np.concatenate(([0.0], np.cumsum(x, dtype=float))), np.concatenate(([0.0], np.cumsum(x * x, dtype=float)))
    Ls = (en-st).astype(np.int64)
    m = (S1[en]-S1[st])/Ls
    m2 = (S2[en]-S2[st])/Ls
    rg = np.sqrt(np.maximum(m2-m*m, 0))
    th = np.median(rg) + 1.4826*np.median(np.abs(rg-np.median(rg)))*float(k_sigma)
    mask = np.interp(np.arange(n), cs, rg) > th
    if pad>0 and mask.any():
        mask = _dilate_mask(mask, fs, pad / fs)
    return mask

def _smooth_binary(m, fs, blend_ms=80):
    L = max(3, int(round((blend_ms / 1000.0) * fs)))
    if L % 2 == 0: L += 1
    w = np.hanning(L); w /= w.sum()
    return np.convolve(m.astype(float), w, mode='same')

try: import pywt
except: pywt = None

@cached_numpy(maxsize=16)
@profiled()
def qrs_aware_wavelet_denoise(y, fs, wavelet='db6', level=None, sigma_scale=2.8, blend_ms=80):
    y_r = np.asarray(y, float)
    N = len(y_r)
    pr = _smooth_binary(make_qrs_mask(y_r, fs), fs, blend_ms)
    try:
        if pywt is None: raise ImportError("pywt not found")
        lv = level or min(5, max(2, int(np.log2(fs/8))))
        cs = pywt.wavedec(y_r, wavelet, level=lv, mode='symmetric')
        cA, ds = cs[0], cs[1:]
        th = float(sigma_scale)*(np.median(np.abs(ds[-1]))/0.6745+1e-12)
        yw = pywt.waverec([cA]+[pywt.threshold(c, th, 'soft') for c in ds], wavelet, 'symmetric')[:N]
    except:
        yw = savgol_filter(y_r, min(N, int(0.05*fs)|1), 2, mode='interp')
    return pr*yw + (1-pr)*y_r


if __name__ == "__main__":
    print("[Standalone ECG Module] Loaded successfully.")
    
    # Generate synthetic signal for benchmark
    fs = 250
    t = np.linspace(0, 60, 60*fs) # 60 seconds
    baseline = 0.5 * np.sin(2 * np.pi * 0.2 * t) + 1.5 * np.exp(-((t-30)/5)**2) # Sine + Sag
    x = baseline + 0.1 * np.random.randn(len(t)) # Noise
    # Add fake R-peaks
    for i in range(0, len(t), 200):
        x[i] += 2.0
        
    print(f"Signal size: {x.size} samples ({x.size/fs:.2f}s)")
    
    # Run core functions multiple times to get stable averages
    iters = 2
    print(f"Running benchmarks ({iters} iterations)...")
    
    for i in range(iters):
        print(f"Iteration {i+1}...")
        
        print("  Running baseline_hybrid_plus_adaptive...")
        y_corr, bl = baseline_hybrid_plus_adaptive(x, fs)
        
        print("  Running masks...")
        suppress_negative_sag(y_corr, fs)
        fix_downward_steps_mask(y_corr, fs)
        smooth_corners_mask(y_corr, fs)
        burst_mask(y_corr, fs)
        high_variance_mask(y_corr)
        
        print("  Running denoise...")
        qrs_aware_wavelet_denoise(y_corr, fs)
    
    print("Benchmark complete.")
    profiler_report()
