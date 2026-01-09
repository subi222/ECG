# train_FCN_DAE.py

import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Lambda, Conv2DTranspose
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import losses
from sklearn.model_selection import train_test_split

# ==========================================================
# 1. 유틸리티 함수 및 손실 함수 (Loss Functions)
# ==========================================================

# 1D Upsampling을 위한 커스텀 레이어
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, activation='relu', padding='same'):
    """
    Keras에 없는 Conv1DTranspose를 Conv2DTranspose로 구현
    """
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=(kernel_size, 1),
                        activation=activation,
                        strides=(strides, 1),
                        padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

# FCN-DAE 전용 손실 함수 (Sum of Squared Distance)
def ssd_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-2)

# ==========================================================
# 2. 모델 정의 (FCN-DAE Model Definition)
# ==========================================================

def build_FCN_DAE(signal_size=512):
    """
    FCN-DAE 모델 구조 정의
    """
    input_shape = (signal_size, 1)
    input_layer = Input(shape=input_shape)

    # --- Encoder ---
    x = Conv1D(filters=40, kernel_size=16, strides=2, padding='same', activation='elu')(input_layer)
    x = BatchNormalization()(x)
    
    x = Conv1D(filters=20, kernel_size=16, strides=2, padding='same', activation='elu')(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(filters=20, kernel_size=16, strides=2, padding='same', activation='elu')(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(filters=20, kernel_size=16, strides=2, padding='same', activation='elu')(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(filters=40, kernel_size=16, strides=2, padding='same', activation='elu')(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(filters=1, kernel_size=16, strides=1, padding='same', activation='elu')(x)
    x = BatchNormalization()(x)

    # --- Decoder ---
    x = Conv1DTranspose(x, filters=1, kernel_size=16, strides=1, padding='same', activation='elu')
    x = BatchNormalization()(x)
    
    x = Conv1DTranspose(x, filters=40, kernel_size=16, strides=2, padding='same', activation='elu')
    x = BatchNormalization()(x)
    
    x = Conv1DTranspose(x, filters=20, kernel_size=16, strides=2, padding='same', activation='elu')
    x = BatchNormalization()(x)
    
    x = Conv1DTranspose(x, filters=20, kernel_size=16, strides=2, padding='same', activation='elu')
    x = BatchNormalization()(x)
    
    x = Conv1DTranspose(x, filters=20, kernel_size=16, strides=2, padding='same', activation='elu')
    x = BatchNormalization()(x)
    
    x = Conv1DTranspose(x, filters=40, kernel_size=16, strides=2, padding='same', activation='elu')
    x = BatchNormalization()(x)

    # Output Layer (Linear Activation)
    predictions = Conv1DTranspose(x, filters=1, kernel_size=16, strides=1, padding='same', activation='linear')

    model = Model(inputs=[input_layer], outputs=predictions)
    return model

# ==========================================================
# 3. 학습 파이프라인 (Training Pipeline)
# ==========================================================

def train_fcn_dae(X_data, y_data, signal_size=512, batch_size=128, epochs=1000):
    
    print('Starting FCN-DAE Training Pipeline...')
    
    # 1. 데이터 분할 (Train : Validation = 7:3)
    # 입력받은 데이터를 자동으로 학습용과 검증용으로 나눕니다.
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.3, shuffle=True, random_state=42)
    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # 2. 모델 불러오기
    model = build_FCN_DAE(signal_size=signal_size)
    model.summary()
    
    # 3. 학습 설정
    lr = 1e-3
    minimum_lr = 1e-10
    
    # FCN-DAE는 SSD Loss를 사용합니다.
    model.compile(loss=ssd_loss,
                  optimizer=keras.optimizers.Adam(lr=lr),
                  metrics=[losses.mean_squared_error, losses.mean_absolute_error, ssd_loss])

    # 4. 콜백(Callbacks) 설정
    model_label = 'FCN_DAE'
    model_filepath = f'{model_label}_weights.best.hdf5'

    # Best Weight 저장
    checkpoint = ModelCheckpoint(model_filepath,
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)

    # 학습률 자동 감소
    reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                  factor=0.5,
                                  min_delta=0.05,
                                  mode='min',
                                  patience=2,
                                  min_lr=minimum_lr,
                                  verbose=1)

    # 조기 종료 (Early Stopping)
    early_stop = EarlyStopping(monitor="val_loss",
                               min_delta=0.05,
                               mode='min',
                               patience=10,
                               verbose=1)

    # 텐서보드 로그
    tb_log_dir = './runs/' + model_label
    tboard = TensorBoard(log_dir=tb_log_dir, histogram_freq=0, write_graph=False, write_images=False)

    # 5. 학습 시작 (Fit)
    history = model.fit(x=X_train, y=y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[early_stop, reduce_lr, checkpoint, tboard])
    
    K.clear_session()
    print("Training Finished. Best weights saved to:", model_filepath)
    return history

# ==========================================================
# 4. 실행 예시 (Main)
# ==========================================================

if __name__ == "__main__":
    # -------------------------------------------------------
    # 사용 예시: .npz 파일 로드 후 학습 실행
    # -------------------------------------------------------
    
    # (예시) 데이터 로드 - 사용자의 데이터 경로로 수정하세요
    # data = np.load('your_data.npz')
    # X_all = data['x_train']  # 노이즈 신호
    # y_all = data['y_train']  # 깨끗한 신호 (정답)

    # (테스트용) 가짜 데이터 생성 (실제 사용 시 삭제하세요)
    print("Generating dummy data for testing...")
    X_all = np.random.rand(1000, 512, 1) # 1000개의 샘플, 길이 512, 채널 1
    y_all = np.random.rand(1000, 512, 1)

    # 학습 함수 호출
    train_fcn_dae(X_all, y_all, signal_size=512)