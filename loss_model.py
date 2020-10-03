import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, MaxPooling2D
from tensorflow.keras import backend as K


def create_model(input_shape=(256, 256, 3), coef=1., alpha=1):
    vgg = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    vgg = Model(inputs=vgg.inputs, outputs=vgg.get_layer('block4_conv1').output, name='vgg')

    content_input = Input(shape=input_shape, name='content_input')
    style_input = Input(shape=input_shape, name='style_input')

    style_out = []
    enc_layers = []
    c = content_input
    s = style_input
    for layer in vgg.layers[1:]:
        if 'conv' in layer.name:
            srp = Padding()
            enc_layers.append(srp)
            c = srp(c)
            s = srp(s)
            new_layer = Conv2D(filters=layer.filters, kernel_size=layer.kernel_size, activation=layer.activation, padding='valid', name=layer.name)
        elif 'pool' in layer.name:
            new_layer = MaxPooling2D((2, 2), strides=(2, 2), name=layer.name)
        else:
            assert False

        enc_layers.append(new_layer)
        c = new_layer(c)
        s = new_layer(s)
        new_layer.set_weights(layer.get_weights())
        
        if 'conv1' in s.name:
            style_out.append(s)

    adain = AdaIN(alpha=alpha, name='adain')([c, s])
    x = adain
    
    # Decoder
    decoder_layers = [
        # Block 4
        Padding(),
        Conv2D(256, (3, 3), activation='relu', padding='valid', name='block4_conv1_decoded'),
        UpSampling2D(),

        # Block 3
        Padding(),
        Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv4_decoded'),
        Padding(),
        Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv3_decoded'),
        Padding(),
        Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv2_decoded'),
        Padding(),
        Conv2D(128, (3, 3), activation='relu', padding='valid', name='block3_conv1_decoded'),
        UpSampling2D(),

        # Block 2
        Padding(),
        Conv2D(128, (3, 3), activation='relu', padding='valid', name='block2_conv2_decoded'),
        Padding(),
        Conv2D(64, (3, 3), activation='relu', padding='valid', name='block2_conv1_decoded'),
        UpSampling2D(),

        # Block 1
        Padding(),
        Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv2_decoded'),
        Padding(),
        Conv2D(3, (3, 3), activation=None, padding='valid', name='block1_conv1_decoded'),

        PostProcess(name="decoded"),
    ]

    for layer in decoder_layers:
        x = layer(x)
          
    # Connections for calculating of losses
    out = []
    for layer in enc_layers:
        x = layer(x)
        if 'conv1' in x.name:
            out.append(x)
    
    loss_model = Model(inputs=[content_input, style_input], outputs=x)
    
    # Content loss
    Lc = tf.reduce_mean(tf.square(adain - x), axis=(1,2,3))
    loss_model.add_loss(Lc)

    # Style loss
    L1 = tf.constant(0.)
    L2 = tf.constant(0.)
    for t, s in zip(out, style_out):
        mean_t, variance_t = tf.nn.moments(t, [1,2])
        mean_s, variance_s = tf.nn.moments(s, [1,2])
        std_t, std_s = tf.sqrt(variance_t), tf.sqrt(variance_s)
        #std_t, std_s = variance_t, variance_s
        L1 += tf.reduce_mean(K.square(mean_t - mean_s), axis=1)
        L2 += tf.reduce_mean(K.square(std_t - std_s), axis=1)
    
    Ls = L1 + L2
    loss_model.add_loss(coef*Ls)

    loss_model.add_metric(Lc, name="Lc")
    loss_model.add_metric(Ls, name="Ls")

    # Weights freezing
    for layer in loss_model.layers:
        layer.trainable = layer.name.endswith('decoded')
    
    return loss_model
