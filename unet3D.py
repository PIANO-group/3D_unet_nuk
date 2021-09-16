import tensorflow as tf

output_channels = 1
batch_norm_before_act = True
batch_norm_after_act = False
l2_reg = 0.001
regularizer = None
# regularizer = tf.keras.regularizers.l2(l=l2_reg)
dropout = 0.5


def conv3d_block(input_tensor, n_filters, kernel_size=3):
    x = input_tensor
    for i in range(2):
        x = tf.keras.layers.Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size), \
                                   kernel_initializer='he_normal', kernel_regularizer=regularizer, padding='same')(x)
        if batch_norm_before_act:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        if batch_norm_after_act:
            x = tf.keras.layers.BatchNormalization()(x)

    return x


def make_encoder_block(inputs, n_filters=64, pool_size=(2, 2, 2), dropout=dropout):
    for_conc = conv3d_block(inputs, n_filters=n_filters)
    enc_block = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(for_conc)
    enc_block = tf.keras.layers.Dropout(dropout)(enc_block)

    return for_conc, enc_block


def make_encoder(inputs):
    for_conc1, enc_block1 = make_encoder_block(inputs, n_filters=64, pool_size=(2, 2, 2), dropout=dropout)
    for_conc2, enc_block2 = make_encoder_block(enc_block1, n_filters=128, pool_size=(2, 2, 2), dropout=dropout)
    for_conc3, enc_block3 = make_encoder_block(enc_block2, n_filters=256, pool_size=(2, 2, 2), dropout=dropout)
    for_conc4, enc_block4 = make_encoder_block(enc_block3, n_filters=512, pool_size=(2, 2, 2), dropout=dropout)

    return enc_block4, (for_conc1, for_conc2, for_conc3, for_conc4)


def make_bottleneck(inputs):
    bottleneck = conv3d_block(inputs, n_filters=1024)

    return bottleneck


def make_decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=dropout):
    upsampled = tf.keras.layers.Conv3DTranspose(n_filters, kernel_size, strides=strides, padding='same')(inputs)
    dec_block = tf.keras.layers.concatenate([upsampled, conv_output])
    dec_block = tf.keras.layers.Dropout(dropout)(dec_block)
    dec_block = conv3d_block(dec_block, n_filters, kernel_size=3)
    return dec_block


def make_decoder(inputs, convs, output_channels):
    for_conc1, for_conc2, for_conc3, for_conc4 = convs

    dec_block4 = make_decoder_block(inputs, for_conc4, n_filters=512, kernel_size=(3, 3, 3), strides=(2, 2, 2), dropout=dropout)
    dec_block3 = make_decoder_block(dec_block4, for_conc3, n_filters=256, kernel_size=(3, 3, 3), strides=(2, 2, 2), dropout=dropout)
    dec_block2 = make_decoder_block(dec_block3, for_conc2, n_filters=128, kernel_size=(3, 3, 3), strides=(2, 2, 2), dropout=dropout)
    dec_block1 = make_decoder_block(dec_block2, for_conc1, n_filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), dropout=dropout)

    outputs = tf.keras.layers.Conv3D(output_channels, (1, 1, 1), activation='sigmoid')(dec_block1)

    return outputs


def unet3D():
    inputs = tf.keras.layers.Input(shape=(64, 64, 32, 1))
    encoder, for_concs = make_encoder(inputs)
    bottleneck = make_bottleneck(encoder)
    outputs = make_decoder(bottleneck, for_concs, output_channels=output_channels)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model