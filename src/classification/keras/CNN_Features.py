from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def build_feature_cnn_model(input_shape, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Block 1
    model.add(Conv1D(64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    # Block 2
    model.add(Conv1D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # Block 3
    model.add(Conv1D(256, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    return model


# Learning rate scheduler
def get_lr_scheduler():
    return ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Optional: Early stopping (you can add this to your callbacks in train.py!)
def get_early_stopping():
    return EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)