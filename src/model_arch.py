from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization

def build_hybrid_model(input_shape, num_classes):
    model = Sequential([
        # CNN Stage
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # LSTM Stage
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        
        # Output Stage
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy', 
                  metrics=['accuracy'])
    return model