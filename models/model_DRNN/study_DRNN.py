"""
DRNN (Deep Recurrent Neural Network) for ECG Signal Denoising

Based on: "Deep Recurrent Neural Networks for ECG Signal Denoising"
- Section 3.1: Network architectures comparison

Best performing architecture:
- LSTM Layer (64 units, return_sequences=True)
- Dense Layer (64 units, ReLU)
- Dense Layer (64 units, ReLU)
- Output Layer (1 unit, Linear)

Key Points:
- Uses return_sequences=True to maintain time dimension
- TimeDistributed wrapper for Dense layers to apply same weights across all time steps
- MSE loss with Adam optimizer (Section 2.4)
"""

import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, TimeDistributed


def build_DRNN(signal_size=512, lstm_units=64, dense_units=64):
    """
    Build DRNN model for ECG denoising.
    
    Architecture (from paper Section 3.1):
    - Input: (signal_size, 1)
    - LSTM: 64 units, return_sequences=True
    - Dense: 64 units, ReLU (TimeDistributed)
    - Dense: 64 units, ReLU (TimeDistributed)
    - Output: 1 unit, Linear (TimeDistributed)
    
    Parameters
    ----------
    signal_size : int
        Length of input signal (default: 512)
    lstm_units : int
        Number of LSTM units (default: 64)
    dense_units : int
        Number of Dense layer units (default: 64)
    
    Returns
    -------
    keras.Model
        Compiled DRNN model
    """
    # Input Layer
    input_layer = Input(shape=(signal_size, 1), name='input')
    
    # LSTM Layer (return_sequences=True to output sequence)
    # Output shape: (batch, signal_size, lstm_units)
    x = LSTM(units=lstm_units, return_sequences=True, name='lstm')(input_layer)
    
    # Dense Layer 1 (TimeDistributed to apply across all time steps)
    # Output shape: (batch, signal_size, dense_units)
    x = TimeDistributed(Dense(dense_units, activation='relu'), name='dense_1')(x)
    
    # Dense Layer 2
    # Output shape: (batch, signal_size, dense_units)
    x = TimeDistributed(Dense(dense_units, activation='relu'), name='dense_2')(x)
    
    # Output Layer (Linear activation for regression)
    # Output shape: (batch, signal_size, 1)
    output_layer = TimeDistributed(Dense(1, activation='linear'), name='output')(x)
    
    # Build Model
    model = Model(inputs=input_layer, outputs=output_layer, name='DRNN')
    
    # Compile (Section 2.4: Adam optimizer, MSE loss)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# ============================================================================
# Test/Demo
# ============================================================================
if __name__ == '__main__':
    import numpy as np
    
    # Build model
    model = build_DRNN(signal_size=512)
    
    # Print summary
    print("\n" + "="*60)
    print("DRNN Model Summary")
    print("="*60)
    model.summary()
    
    # Test forward pass
    print("\n" + "="*60)
    print("Forward Pass Test")
    print("="*60)
    
    dummy_input = np.random.randn(4, 512, 1).astype(np.float32)
    dummy_output = model.predict(dummy_input, verbose=0)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
    print(f"Expected:     (4, 512, 1)")
    
    if dummy_output.shape == (4, 512, 1):
        print("\n✅ Shape verification PASSED!")
    else:
        print("\n❌ Shape verification FAILED!")
