import numpy as np
import tensorflow as tf

def build_model(arch, dropout_rate, lr, l2_weight, vocab_size, seq_len):
    reg = tf.keras.regularizers.l2(l2_weight) if l2_weight > 0 else None
    embedding_size = int(np.sqrt(vocab_size))

    tokens = tf.keras.Input(shape=(seq_len,), dtype='int64')
    mask = tf.keras.ops.not_equal(tokens, 0)
    embeddings = tf.keras.layers.Embedding(vocab_size, embedding_size)(tokens)
    embeddings = tf.keras.layers.Dropout(dropout_rate)(embeddings)

    # conv layer
    query = tf.keras.layers.Conv1D(arch['filters'], arch['kernel_size'], padding='same', kernel_regularizer=reg, name='query')(embeddings)
    value = tf.keras.layers.Conv1D(arch['filters'], arch['kernel_size'], padding='same', kernel_regularizer=reg, name='value')(embeddings)

    # padding mask for all models
    attention_mask = tf.keras.ops.logical_and(tf.keras.ops.expand_dims(mask, axis=2),
                                              tf.keras.ops.expand_dims(mask, axis=1))

    if arch['type'] == 'transformer':
        attention = tf.keras.layers.MultiHeadAttention(num_heads=arch['heads'], key_dim=arch['filters'] // arch['heads'],
                                                       dropout=dropout_rate, name='attention')(query, value, attention_mask=attention_mask)
        attention = tf.keras.layers.Dropout(dropout_rate)(attention)
        x = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([query, attention]))
        ffn = tf.keras.layers.Dense(arch['ff_dim'], activation='relu', kernel_regularizer=reg)(x)
        ffn = tf.keras.layers.Dense(arch['filters'], kernel_regularizer=reg)(ffn)
        ffn = tf.keras.layers.Dropout(dropout_rate)(ffn)
        x = tf.keras.layers.LayerNormalization()(tf.keras.layers.Add()([x, ffn]))
        x = tf.keras.layers.GlobalAveragePooling1D()(x, mask=mask)
    else:
        attention = tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=arch['filters'],
                                                       dropout=dropout_rate, name='attention')(query, value, attention_mask=attention_mask)
        x = tf.keras.layers.Concatenate()([query, attention])
        rnn = tf.keras.layers.LSTM if arch['type'] == 'lstm' else tf.keras.layers.GRU
        for i, units in enumerate(arch['units']):
            last = i == len(arch['units']) - 1
            x = rnn(units, return_sequences=not last, kernel_regularizer=reg)(x, mask=mask)

    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(tokens, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0), loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')])
    return model
