import tensorflow as tf

def build_model(arch, dropout_rate, lr, l2_weight=1e-4, augment=True):
    reg = tf.keras.regularizers.l2(l2_weight) if l2_weight > 0 else None

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(32, 32, 3)))
    model.add(tf.keras.layers.Rescaling(scale=1./255))

    if augment:
        model.add(tf.keras.layers.RandomFlip('horizontal'))
        model.add(tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='reflect'))

    for filters in arch['filters']:
        for _ in range(arch['convs_per_block']):
            model.add(tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_regularizer=reg, use_bias=not arch['batch_norm']))
            if arch['batch_norm']:
                model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Flatten())

    for size in arch['dense']:
        model.add(tf.keras.layers.Dense(size, tf.nn.relu, kernel_regularizer=reg))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(100, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')])
    return model
