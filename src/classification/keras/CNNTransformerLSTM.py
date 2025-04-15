from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense, Flatten

def build_cnn_bilstm_model(input_shape, num_classes=2):
    inputs = Input(shape=input_shape)

    # === CNN part ===
    x = Conv1D(filters=64, kernel_size=7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=128, kernel_size=5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Dropout(0.3)(x)

    # === BiLSTM part ===
    x = Bidirectional(LSTM(64, return_sequences=False))(x)

    # === Fully connected ===
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes)(x)  # no activation, because from_logits=True

    model = Model(inputs, outputs)

    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



import tensorflow as tf

def positional_encoding(x):
    seq_len = tf.shape(x)[1]  # dynamic shape
    d_model = tf.shape(x)[2]

    position = tf.cast(tf.range(seq_len)[:, tf.newaxis], dtype=tf.float32)
    div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / tf.cast(d_model, tf.float32)))

    angle_rates = tf.matmul(position, div_term[tf.newaxis, :])

    # Create sin and cos positional encodings
    sines = tf.sin(angle_rates)
    cosines = tf.cos(angle_rates)

    # Concatenate along last axis
    pos_encoding = tf.concat([sines, cosines], axis=-1)

    # Expand batch dimension to broadcast correctly
    pos_encoding = tf.expand_dims(pos_encoding, axis=0)

    return pos_encoding
