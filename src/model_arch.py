from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization

def build_hybrid_model(input_shape=(52, 1), num_classes=2):
    """
    Builds a Hybrid 1D-CNN + LSTM Neural Network for Network Intrusion Detection (NIDS).
    
    Parameters:
    - input_shape: tuple, shape of input sequence (features, channels). Default: (52, 1).
    - num_classes: int, number of target classes (2 for Binary, >2 for Multi-class).
    
    Returns:
    - Compiled Keras Sequential model.
    """
    is_multiclass = num_classes > 2
    
    model = Sequential([
        # 1D-CNN Stage: Spatial Feature Extraction
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # LSTM Stage: Temporal Pattern Learning
        LSTM(units=64, return_sequences=False),
        Dropout(0.3),
        
        # Dense Classification Stage
        Dense(32, activation='relu'),
        Dense(
            units=num_classes if is_multiclass else 1, 
            activation='softmax' if is_multiclass else 'sigmoid'
        )
    ])
    
    # Select loss function based on classification type
    loss_fn = 'sparse_categorical_crossentropy' if is_multiclass else 'binary_crossentropy'
    
    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    return model