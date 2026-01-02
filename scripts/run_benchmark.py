# scripts/run_benchmark.py
import argparse
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional


ROOT = Path(__file__).resolve().parents[1]  # scripts/ 의 상위 = 프로젝트 루트
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ 자체도 import 가능

from benchmark_core import BenchmarkArgs, RunContext, RunnerFn, run_benchmark

# 우리 알고리즘 (v37)
from models.model_proposed.v37_standalone import v37_baseline_correction
# 타모델
from models.model_DAE.model_DAE import ImprovedDAE
from models.model_UNet import UNet


# -------------------------
# Proposed runner (v37)
# -------------------------
def run_method_proposed(x_in: np.ndarray, ctx: RunContext) -> np.ndarray:
    # v37_standalone의 엔트리 함수 그대로 사용 :contentReference[oaicite:4]{index=4}
    y, _baseline = v37_baseline_correction(
        x_in,
        fs=ctx.fs,
        r_idx=ctx.r_idx,
        adaptive_denoise=True
    )
    return y.astype(np.float32)


# -------------------------
# Improved DAE runner
# -------------------------
_DAE_MODEL: Optional[ImprovedDAE] = None
_DAE_DEVICE: Optional[torch.device] = None


def _load_dae_model(ckpt_path: Path, device: str = "cpu") -> ImprovedDAE:
    """
    ckpt 로딩 규칙:
    - torch.save(model.state_dict()) 형태면 state_dict 로드
    - torch.save({"model": state_dict, ...}) 형태면 내부 키 탐색
    """
    global _DAE_MODEL, _DAE_DEVICE
    _DAE_DEVICE = torch.device(device)

    model = ImprovedDAE(window_len=101)
    obj = torch.load(str(ckpt_path), map_location=_DAE_DEVICE)

    if isinstance(obj, dict):
        if "state_dict" in obj:
            sd = obj["state_dict"]
        elif "model" in obj:
            sd = obj["model"]
        else:
            # state_dict처럼 보이는 dict일 수도 있음
            sd = obj
    else:
        sd = obj

    # DataParallel/Lightning prefix 처리
    sd2 = {}
    for k, v in sd.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]
        if kk.startswith("net."):
            # model_DAE.ImprovedDAE 는 self.net 안에 Sequential을 들고 있으니
            # 저장 구조에 따라 net. prefix가 있을 수 있음
            pass
        sd2[kk] = v

    model.load_state_dict(sd2, strict=False)
    model.to(_DAE_DEVICE).eval()
    _DAE_MODEL = model
    return model


def _dae_denoise_fullsignal(
    x: np.ndarray,
    model: ImprovedDAE,
    device: torch.device,
    window_len: int = 101,
    stride: int = 1,
    batch_size: int = 512,
) -> np.ndarray:
    """
    model_DAE 문서에 맞춰:
    - 윈도우 단위로 [0,1] min-max 정규화 (윈도우별)
    - 출력도 [0,1]이므로 같은 min/max로 역정규화
    - overlap-add 평균으로 전체 시계열 복원
    :contentReference[oaicite:5]{index=5}
    """
    x = np.asarray(x, dtype=np.float32)
    N = x.size
    if N == 0:
        return x

    radius = window_len // 2
    x_pad = np.pad(x, (radius, radius), mode="reflect")
    Np = x_pad.size

    # 시작 인덱스들 (pad 기준)
    starts = np.arange(0, Np - window_len + 1, stride, dtype=np.int64)

    out_sum = np.zeros(Np, dtype=np.float32)
    out_cnt = np.zeros(Np, dtype=np.float32)

    # 배치 처리
    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            s_batch = starts[i:i + batch_size]
            # (B, 101)
            windows = np.stack([x_pad[s:s + window_len] for s in s_batch], axis=0).astype(np.float32)

            w_min = windows.min(axis=1, keepdims=True)
            w_max = windows.max(axis=1, keepdims=True)
            denom = (w_max - w_min)
            denom_safe = np.where(denom < 1e-8, 1.0, denom)

            w_norm = (windows - w_min) / denom_safe
            w_norm = np.clip(w_norm, 0.0, 1.0)

            inp = torch.from_numpy(w_norm).to(device=device, dtype=torch.float32)
            pred = model(inp).detach().cpu().numpy().astype(np.float32)

            # 역정규화
            pred_denorm = pred * denom_safe + w_min

            # overlap-add
            for j, s in enumerate(s_batch):
                out_sum[s:s + window_len] += pred_denorm[j]
                out_cnt[s:s + window_len] += 1.0

    out = out_sum / np.maximum(out_cnt, 1.0)
    # pad 제거 -> 원래 길이
    out = out[radius:radius + N]
    return out.astype(np.float32)


def run_method_dae(x_in: np.ndarray, ctx: RunContext) -> np.ndarray:
    """
    ctx.device는 benchmark_core에서 넘김.
    ckpt_path는 전역 ARGS에서 설정(아래 parse_args 참고)
    """
    if ARGS.dae_ckpt is None:
        raise ValueError("DAE method selected but --dae_ckpt is not provided.")

    global _DAE_MODEL, _DAE_DEVICE
    if _DAE_MODEL is None:
        _load_dae_model(Path(ARGS.dae_ckpt), device=ctx.device or ARGS.device)

    return _dae_denoise_fullsignal(
        x_in,
        model=_DAE_MODEL,
        device=_DAE_DEVICE,
        window_len=101,
        stride=ARGS.dae_stride,
        batch_size=ARGS.dae_batch,
    )

# -------------------------
# UNet 1D runner (Universal multi-SNR trained)
# -------------------------
_UNET_MODEL: Optional[UNet] = None
_UNET_DEVICE: Optional[torch.device] = None


def _load_unet_model(ckpt_path: Path, device: str = "cpu") -> UNet:
    """
    ckpt 로딩 규칙:
    - torch.save(model.state_dict()) 형태면 state_dict 로드
    - torch.save({"state_dict":..., ...}) 형태면 내부 키 탐색
    """
    global _UNET_MODEL, _UNET_DEVICE
    _UNET_DEVICE = torch.device(device)

    model = UNet(in_channels=1, out_classes=1, dimensions=1, padding=True)
    obj = torch.load(str(ckpt_path), map_location=_UNET_DEVICE)

    if isinstance(obj, dict):
        if "state_dict" in obj:
            sd = obj["state_dict"]
        elif "model" in obj:
            sd = obj["model"]
        else:
            sd = obj
    else:
        sd = obj

    # DataParallel prefix 처리
    sd2 = {}
    for k, v in sd.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]
        sd2[kk] = v

    model.load_state_dict(sd2, strict=False)
    model.to(_UNET_DEVICE).eval()
    _UNET_MODEL = model
    return model


def _unet_denoise_fullsignal(
    x: np.ndarray,
    model: UNet,
    device: torch.device,
    win_len: int = 512,
    hop_len: int = 512,
    batch_size: int = 64,
    normalize: str = "minmax_by_noisy",
) -> np.ndarray:
    """
    train_unet.py와 동일한 윈도우/정규화 규칙으로 추론:
      - normalize="minmax_by_noisy":
          x_win min/max로 x_norm, y_norm 둘 다 같은 denom 사용
          pred는 y_norm이므로 같은 min/max로 역정규화
      - overlap-add 평균으로 full signal 복원
    """
    x = np.asarray(x, dtype=np.float32)
    N = x.size
    if N == 0:
        return x

    if hop_len <= 0:
        raise ValueError("hop_len must be positive.")
    if win_len <= 0:
        raise ValueError("win_len must be positive.")
    if N < win_len:
        # 짧으면 reflect pad로 1윈도우 처리
        pad = win_len - N
        x_pad = np.pad(x, (0, pad), mode="reflect")
        Np = x_pad.size
    else:
        x_pad = x
        Np = N

    # 시작 인덱스 (마지막이 win_len을 넘지 않도록)
    starts = np.arange(0, Np - win_len + 1, hop_len, dtype=np.int64)

    out_sum = np.zeros(Np, dtype=np.float32)
    out_cnt = np.zeros(Np, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            s_batch = starts[i:i + batch_size]
            # (B, 1, L)
            windows = np.stack([x_pad[s:s + win_len] for s in s_batch], axis=0).astype(np.float32)
            windows = windows[:, None, :]  # add channel dim

            if normalize == "minmax_by_noisy":
                # per-window min/max (batch, 1, 1) for broadcasting
                w_min = windows.min(axis=2, keepdims=True)
                w_max = windows.max(axis=2, keepdims=True)
                denom = (w_max - w_min)
                denom_safe = np.where(denom < 1e-8, 1.0, denom)

                w_norm = (windows - w_min) / denom_safe
                w_norm = np.clip(w_norm, 0.0, 1.0)

                inp = torch.from_numpy(w_norm).to(device=device, dtype=torch.float32)
                pred_norm = model(inp).detach().cpu().numpy().astype(np.float32)  # (B,1,L)

                # 역정규화: y = y_norm * denom + x_min
                pred = pred_norm * denom_safe + w_min

            elif normalize == "none":
                inp = torch.from_numpy(windows).to(device=device, dtype=torch.float32)
                pred = model(inp).detach().cpu().numpy().astype(np.float32)
            else:
                raise ValueError("normalize must be 'minmax_by_noisy' or 'none'")

            # overlap-add
            for j, s in enumerate(s_batch):
                out_sum[s:s + win_len] += pred[j, 0, :]
                out_cnt[s:s + win_len] += 1.0

    out = out_sum / np.maximum(out_cnt, 1.0)

    # 원래 길이로 자르기(짧은 입력 pad 케이스 포함)
    out = out[:N]
    return out.astype(np.float32)


def run_method_unet(x_in: np.ndarray, ctx: RunContext) -> np.ndarray:
    """
    Universal UNet:
    - train_unet.py 결과 best_model.pth 로드
    - 윈도우 기반 overlap-add로 full signal denoise
    """
    if ARGS.unet_ckpt is None:
        raise ValueError("UNet method selected but --unet_ckpt is not provided.")

    global _UNET_MODEL, _UNET_DEVICE
    if _UNET_MODEL is None:
        _load_unet_model(Path(ARGS.unet_ckpt), device=ctx.device or ARGS.device)

    return _unet_denoise_fullsignal(
        x_in,
        model=_UNET_MODEL,
        device=_UNET_DEVICE,
        win_len=ARGS.unet_win_len,
        hop_len=ARGS.unet_hop_len,
        batch_size=ARGS.unet_batch,
        normalize=ARGS.unet_normalize,
    )

# -------------------------
# Registry
# -------------------------
def build_method_registry() -> Dict[str, RunnerFn]:
    return {
        "proposed": run_method_proposed,
        "dae": run_method_dae,
        "unet": run_method_unet,
        # unet1d 등 다른 모델은 나중에 추가
    }


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_path", type=str, default=str(ROOT / "common" / "splits.json"))
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--methods", type=str, default="proposed", help="comma separated: proposed,dae")
    ap.add_argument("--noise_rec", type=str, default="bw", choices=["bw", "ma", "em"])
    ap.add_argument("--snrs", type=str, default="0,5,10,15")
    ap.add_argument("--fs_target", type=float, default=250.0)
    ap.add_argument("--start_sec", type=int, default=0)
    ap.add_argument("--duration_sec", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default=str(Path("outputs/benchmark_test")))
    ap.add_argument("--plot_one", action="store_true")
    ap.add_argument("--plot_rec", type=str, default="")
    ap.add_argument("--preset", type=str, default="", help="paper | debug")

    # device
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # DAE options
    ap.add_argument("--dae_ckpt", type=str, default=None, help="path to trained DAE .pth/.pt")
    ap.add_argument("--dae_stride", type=int, default=1, help="window stride (1=best, larger=faster)")
    ap.add_argument("--dae_batch", type=int, default=512, help="DAE inference batch size")

    # UNet options
    ap.add_argument("--unet_ckpt", type=str, default=None, help="path to trained UNet best_model.pth")
    ap.add_argument("--unet_win_len", type=int, default=512, help="UNet inference window length (must match training)")
    ap.add_argument("--unet_hop_len", type=int, default=512, help="UNet inference hop length (overlap if < win_len)")
    ap.add_argument("--unet_batch", type=int, default=64, help="UNet inference batch size")
    ap.add_argument("--unet_normalize", type=str, default="minmax_by_noisy", choices=["minmax_by_noisy", "none"])


    return ap.parse_args()

def main():
    global ARGS
    ARGS = parse_args()
    ARGS = apply_preset(ARGS)   # ✅ preset 먼저 적용

    snr_levels = [float(s) for s in ARGS.snrs.split(",") if s.strip() != ""]
    method_names = [m.strip() for m in ARGS.methods.split(",") if m.strip() != ""]

    cfg = BenchmarkArgs(
        split_path=Path(ARGS.split_path),
        split=ARGS.split,
        methods=method_names,
        noise_rec=ARGS.noise_rec,
        snr_levels=snr_levels,
        fs_target=float(ARGS.fs_target),
        start_sec=int(ARGS.start_sec),
        duration_sec=int(ARGS.duration_sec),
        seed=int(ARGS.seed),
        out_dir=Path(ARGS.out_dir),
        plot_one=bool(ARGS.plot_one),
        plot_rec=str(ARGS.plot_rec),
    )

    registry = build_method_registry()
    run_benchmark(cfg, registry)


def apply_preset(args):
    if args.preset == "paper":
        args.methods = "proposed,dae,unet"
        args.snrs = "0,5,10,15"
        args.plot_one = True
        args.plot_rec = ""   # 모든 rec
        args.out_dir = str(ROOT / "outputs" / "paper_benchmark")
        args.dae_ckpt = "/home/subi/PycharmProjects/ECG/outputs/Improved_DAE/best_model.pth"
        args.unet_ckpt = "/home/subi/PycharmProjects/ECG/outputs/UNet/best_model.pth"
        args.unet_win_len = 512
        args.unet_hop_len = 512
        args.unet_normalize = "minmax_by_noisy"

    elif args.preset == "debug":
        args.methods = "proposed"
        args.snrs = "0"
        args.plot_one = True
        args.out_dir = "outputs/debug"

    return args


if __name__ == "__main__":
    main()