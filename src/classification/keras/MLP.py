import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# === MLP model for extracted features ===
def build_mlp_model(input_shape, learning_rate=0.001):
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(2, activation='softmax')  # Keep softmax
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',  # <- updated for soft labels
        metrics=['accuracy']
    )
    return model



# === Callbacks ===
def get_lr_scheduler():
    return ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

def get_early_stopping():
    return EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
