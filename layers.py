import tensorflow as tf
from tensorflow.keras.layers import Layer


class AdaIN(Layer):
    def __init__(self, alpha=1.0, **kwargs):
        self.alpha = alpha
        super(AdaIN, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def call(self, inp):
        c, s = inp[0], inp[1]
        # moments
        c_mean, c_variance = tf.nn.moments(c, [1,2], keepdims=True)
        s_mean, s_variance = tf.nn.moments(s, [1,2], keepdims=True)
        eps = 1e-5

        # rescaling
        rescaled_c = tf.nn.batch_normalization(c, c_mean, c_variance, s_mean, tf.sqrt(s_variance), eps)
        rescaled_c = self.alpha * rescaled_c + (1 - self.alpha) * c
        return rescaled_c

class Padding(Layer):

    def __init__(self, **kwargs):
        super(Padding, self).__init__(**kwargs)

    def call(self, x):
        return tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")

class PostProcess(Layer):
 
    def __init__(self, **kwargs):
        super(PostProcess, self).__init__(**kwargs)
 
    def call(self, x):
        x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0)
        return x