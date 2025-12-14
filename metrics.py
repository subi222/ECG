import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def SNR(y1, y2):
    N = np.sum(np.square(y1), axis=1)
    D = np.sum(np.square(y2 - y1))

    SNR = 10 * np.log10(N / D)

    return SNR


def RMSE(y, y_pred):
    # axis=1 은 신호 길이 방향으로 평균을 낸다는 뜻 (기존 코드와 통일)
    return np.sqrt(np.mean(np.square(y - y_pred)))



def NRMSE(y, y_pred):
    """
    Normalized RMSE (정규화된 RMSE)
    :return: 0~1 사이의 값 (낮을수록 좋음, %로 환산 가능)
    """
    rmse_val = RMSE(y, y_pred)
    # 신호의 범위(Max - Min)로 나누어 정규화
    data_range = np.max(y) - np.min(y)
    return rmse_val / data_range



def SNR_improvement(y_in, y_out, y_clean):
    return SNR(y_clean, y_out) - SNR(y_clean, y_in)


def COS_SIM(y, y_pred):
    cos_sim = []
    y = np.squeeze(y, axis=-1)
    y_pred = np.squeeze(y_pred, axis=-1)
    for idx in range(len(y)):
        kl_temp = cosine_similarity(y[idx].reshape(1, -1), y_pred[idx].reshape(1, -1))
        cos_sim.append(kl_temp)

    cos_sim = np.array(cos_sim)
    return cos_sim


def SSD(y, y_pred):
    return np.sum(np.square(y - y_pred), axis=1)  # axis 1 is the signal dimension


def MAD(y, y_pred):
    return np.max(np.abs(y - y_pred), axis=1)  # axis 1 is the signal dimension


def PRD(y, y_pred):
    N = np.sum(np.square(y_pred - y), axis=1)
    D = np.sum(np.square(y_pred - np.mean(y)), axis=1)

    PRD = np.sqrt(N / D) * 100

    return PRD