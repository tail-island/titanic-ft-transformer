import tensorflow as tf

from funcy import rcompose


def create_model(block_size, d_model, head_size, ffn_factor, attention_dropout, ffn_dropout, x_1_vocab_size):
    class Tokenize(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()

        def build(self, input_shape):
            self.kernel = self.add_weight('kernel', shape=(input_shape[-1], d_model), initializer='he_normal')

        def call(self, inputs):
            return tf.expand_dims(inputs, 2) * self.kernel

        def get_config(self):
            config = super().get_config()
            config.update({"kernel": self.kernel.numpy()})
            return config

    def Add():
        return tf.keras.layers.Add()

    def Dense(units):
        return tf.keras.layers.Dense(units, use_bias=False, kernel_initializer='he_uniform')

    def Dropout(rate):
        return tf.keras.layers.Dropout(rate)

    def Embedding(input_dim):
        return tf.keras.layers.Embedding(input_dim, d_model)

    def FeedForwardNetwork():
        return rcompose(Dense(d_model * ffn_factor),
                        GeLU(),
                        Dense(d_model),
                        Dropout(ffn_dropout))

    def GeLU():
        return tf.keras.activations.gelu

    def LayerNormalization():
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def MultiHeadAttention():
        return tf.keras.layers.MultiHeadAttention(head_size, d_model, dropout=attention_dropout, kernel_initializer='he_normal')

    def SelfAttention():
        def op(inputs):
            return MultiHeadAttention()(inputs, inputs)

        return op

    def Sigmoid():
        return tf.keras.activations.sigmoid

    def op(inputs):
        o_1, o_2 = inputs

        o = tf.concat((Embedding(1)(tf.zeros((tf.shape(o_1)[0], 1))),  # CLS
                       Embedding(x_1_vocab_size)(o_1),                 # カテゴリ値の入力
                       Tokenize()(o_2)),                               # スカラー値の入力
                      axis=1)

        for _ in range(block_size):
            o = Add()((o, SelfAttention()(LayerNormalization()(o))))
            o = Add()((o, FeedForwardNetwork()(LayerNormalization()(o))))

        o = Sigmoid()(Dense(1)(o[:, 0]))

        return o

    return op
