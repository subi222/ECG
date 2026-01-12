import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

def DRRN_denoising(signal_size=512):
    """
    Implementation of DRNN approach presented in:
    Antczak, K. (2018). Deep recurrent neural networks for ECG signal denoising.
    arXiv preprint arXiv:1807.11551.
    """
    model = Sequential()
    
    # 1. LSTM Layer: 시계열 특징 추출 (64 units)
    # return_sequences=True: 모든 타임스텝에 대해 출력을 반환 (Many-to-Many)
    model.add(LSTM(64, input_shape=(signal_size, 1), return_sequences=True))
    
    # 2. Fully Connected Layers: 비선형 변환
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    
    # 3. Output Layer: 최종 신호 출력 (Linear)
    model.add(Dense(1, activation='linear'))

    return model