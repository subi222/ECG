import tensorflow as tf
from tensorflow.keras import layers, models, Input
import math

# ==========================================================
# 1. 유틸리티 레이어 (Half-Instance Norm, Positional Encoding)
# ==========================================================

class HalfInstanceNormalization(layers.Layer):
    """
    HNF Block의 핵심인 Half-Instance Normalization
    입력 채널을 절반으로 나누어, 절반은 Instance Norm을 적용하고 나머지 절반은 그대로 둠.
    """
    def __init__(self, epsilon=1e-5, **kwargs):
        super(HalfInstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # Instance Normalization용 Scale(gamma)과 Shift(beta) 파라미터
        # 절반 채널에 대해서만 적용하므로 shape은 channel // 2
        channels = input_shape[-1]
        self.half_channels = channels // 2
        
        self.gamma = self.add_weight(name='gamma', shape=(self.half_channels,),
                                     initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=(self.half_channels,),
                                    initializer='zeros', trainable=True)
        super(HalfInstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        # 1. 채널 분할
        x1, x2 = tf.split(inputs, num_or_size_splits=2, axis=-1)

        # 2. x1에 대해 Instance Normalization 수행 (Time 축 기준)
        # 1D 신호: (Batch, Length, Channels) -> axes=[1]로 평균/분산 계산
        mean, variance = tf.nn.moments(x1, axes=[1], keepdims=True)
        x1_norm = (x1 - mean) / tf.sqrt(variance + self.epsilon)
        x1_final = x1_norm * self.gamma + self.beta

        # 3. 다시 결합 (정규화된 절반 + 원래 절반)
        return tf.concat([x1_final, x2], axis=-1)


def sinusoidal_time_embedding(time_steps, embedding_dim=128):
    """
    Bridge 블록에 들어가는 사인파 위치 임베딩 (Sinusoidal Positional Embeddings)
    time_steps: (Batch, 1) 형태의 alpha_bar 값 또는 스텝
    """
    # 로그 스케일의 주파수 생성
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    
    # (Batch, 1) * (Half_Dim) -> (Batch, Half_Dim)
    emb = tf.cast(time_steps, dtype=tf.float32) * emb[None, :]
    
    # Sin, Cos 결합 -> (Batch, Dim)
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
    return emb


# ==========================================================
# 2. 핵심 블록 (HNF Block, Bridge Block)
# ==========================================================

def HNF_Block(x, filters=80):
    """
    Figure 2 (B): Half Normalized Filters Block
    - Multi-scale filters -> Concat -> Conv -> Half-IN -> Conv -> Residual
    """
    shortcut = x
    
    # 1. Multi-scale Filters (receptive field 확장)
    # 80채널을 4개로 분할하여 각각 다른 커널 크기 적용 (20채널씩)
    branch_filters = filters // 4
    
    conv_k3  = layers.Conv1D(branch_filters, kernel_size=3, padding='same')(x)
    conv_k15 = layers.Conv1D(branch_filters, kernel_size=15, padding='same')(x)
    conv_k9  = layers.Conv1D(branch_filters, kernel_size=9, padding='same')(x)
    conv_k5  = layers.Conv1D(branch_filters, kernel_size=5, padding='same')(x)
    
    # 2. Concatenate & Aggregate
    concat = layers.Concatenate(axis=-1)([conv_k3, conv_k15, conv_k9, conv_k5]) # Total 80
    x = layers.Conv1D(filters, kernel_size=9, padding='same')(concat)
    
    # 3. Half-Instance Norm branch
    # 구조도 상: Channel Split -> (Inst Norm | No Norm) -> Concat -> LeakyReLU
    x = HalfInstanceNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # 4. Final Conv & Residual
    x = layers.Conv1D(filters, kernel_size=9, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Residual Addition
    output = layers.Add()([shortcut, x])
    
    return output


def Bridge_Block(feature_map, time_emb_vector, filters=80):
    """
    Figure 2 (C): Bridge Block
    - 특징 맵(Right Stream)과 시간 정보(alpha_bar)를 결합
    - Scale & Shift 연산을 통해 시간 정보를 특징 맵에 주입
    """
    # 1. Time Embedding 처리 (Architecture C 왼쪽 부분)
    # 이미 인코딩된 time_emb_vector가 들어온다고 가정 (혹은 여기서 추가 처리)
    # 구조도 상: Positional Encoding -> 1x1 Conv (160) -> Split
    
    # 1x1 Conv와 동일한 Dense 레이어 사용 (입력이 벡터이므로)
    t = layers.Dense(filters * 2)(time_emb_vector) # 160 units
    
    # Channel Split (Scale vector, Shift vector)
    scale, shift = tf.split(t, num_or_size_splits=2, axis=-1)
    
    # (Batch, 1, Filters) 형태로 변환하여 브로드캐스팅 준비
    scale = layers.Reshape((1, filters))(scale)
    shift = layers.Reshape((1, filters))(shift)
    
    # 2. Feature Map 처리 (Architecture C 오른쪽 부분)
    # Input -> 3x1 Conv (80)
    x = layers.Conv1D(filters, kernel_size=3, padding='same')(feature_map)
    
    # 3. Interaction (곱셈과 덧셈)
    x = layers.Multiply()([x, scale])
    x = layers.Add()([x, shift])
    
    # 4. Final Conv
    x = layers.Conv1D(filters, kernel_size=3, padding='same')(x)
    
    return x


# ==========================================================
# 3. 전체 모델 조립 (Architecture A)
# ==========================================================

def build_DeScoD_ECG(signal_length=None, filters=80):
    """
    Figure 2 (A): 전체 아키텍처
    - Left Stream (xt): 잡음이 섞인 잠재 변수 처리 (Denoising path)
    - Right Stream (x_tilde): 조건부 입력 처리 (Condition path)
    - Bridge: Right Stream 정보를 Left Stream으로 주입
    """
    
    # --- Inputs ---
    # 1. xt: Noise가 섞인 Latent Variable (Denoising 대상)
    input_xt = Input(shape=(signal_length, 1), name='input_xt')
    
    # 2. x_tilde: Condition (원본에 가까운 참조 신호 or 관측치)
    input_cond = Input(shape=(signal_length, 1), name='input_condition')
    
    # 3. alpha_bar (or time step): 노이즈 레벨 정보
    input_time = Input(shape=(1,), name='input_time')
    
    
    # --- Positional Encoding (Shared) ---
    # 구조도 C의 상단 부분: sqrt(alpha) -> Positional Encoding
    # 여기서는 input_time을 sqrt(alpha)라고 가정하고 임베딩 생성
    time_emb = layers.Lambda(lambda t: sinusoidal_time_embedding(t, embedding_dim=128))(input_time)
    
    
    # --- Initial Convolution (Entry) ---
    # Left Stream
    x_left = layers.Conv1D(filters, kernel_size=9, padding='same')(input_xt)
    x_left = layers.LeakyReLU(alpha=0.2)(x_left)
    
    # Right Stream
    x_right = layers.Conv1D(filters, kernel_size=9, padding='same')(input_cond)
    x_right = layers.LeakyReLU(alpha=0.2)(x_right)
    
    
    # --- Stacked Blocks (5 stages) ---
    # 구조도 A: 5개의 HNF 블록과 Bridge가 반복됨
    
    for i in range(5):
        # 1. Right Stream (Condition Path) Processing
        # 구조도 상 오른쪽 HNF가 먼저 처리되어 Bridge로 정보를 보냄
        x_right = HNF_Block(x_right, filters=filters)
        
        # 2. Bridge Processing
        # Right Stream의 특징과 시간 정보를 결합하여 Scale/Shift 정보 생성
        bridge_out = Bridge_Block(x_right, time_emb, filters=filters)
        
        # 3. Information Injection (Bridge -> Left Stream)
        # 구조도 A에서 Bridge의 화살표가 Left Stream으로 합류함
        x_left = layers.Add()([x_left, bridge_out])
        
        # 4. Left Stream (Main Path) Processing
        x_left = HNF_Block(x_left, filters=filters)
        
    
    # --- Output Layer ---
    # 구조도 A 상단: 9x1 Conv (1) -> output (predicted noise epsilon)
    output = layers.Conv1D(1, kernel_size=9, padding='same', activation='linear', name='output_noise')(x_left)
    
    
    # --- Model Definition ---
    model = models.Model(inputs=[input_xt, input_cond, input_time], outputs=output, name='DeScoD_ECG')
    
    return model

# ==========================================================
# 4. 모델 생성 예시
# ==========================================================

# 모델 빌드 (신호 길이 512 기준)
model = build_DeScoD_ECG(signal_length=512)
model.summary()