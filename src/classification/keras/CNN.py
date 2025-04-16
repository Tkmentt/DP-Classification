import tensorflow as tf
from keras import Model, Sequential
from keras.constraints import max_norm
from keras.layers import Conv2D, BatchNormalization, Dropout, AveragePooling2D, Flatten, Dense, DepthwiseConv2D, \
    Activation, SeparableConv2D, Conv1D, DepthwiseConv1D, SeparableConv1D, AveragePooling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def build_cnn_model(input_shape, learning_rate=0.001):
    """
    EEGNet-inspired CNN model for raw EEG classification.
    Assumes input shape: (time_steps, channels)
    """

    model = Sequential()

    # Block 1
    model.add(Conv1D(filters=8, kernel_size=64, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(DepthwiseConv1D(kernel_size=64, depth_multiplier=2, depthwise_constraint=max_norm(1.)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling1D(pool_size=4))
    model.add(Dropout(0.5))

    # Block 2
    model.add(SeparableConv1D(filters=16, kernel_size=16, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling1D(pool_size=4))
    model.add(Dropout(0.5))

    # Classification head
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))  # 2 classes (soft labels: [1, 0], [0, 1], or [0.5, 0.5])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Learning rate scheduler
def get_lr_scheduler():
    return ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Optional: Early stopping (you can add this to your callbacks in train.py!)
def get_early_stopping():
    return EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

""" OLD MODEL
def build_cnn_model(input_shape, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Block 1
    model.add(Conv1D(64, kernel_size=5, padding='same'))
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

    # Classification head
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
"""
