import tensorflow as tf

from funcy import rcompose


def create_model(block_size, d_model, head_size, ffn_factor, attention_dropout, ffn_dropout):
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
        x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8 = tf.split(inputs, [1, 1, 1, 1, 1, 1, 1, 1], 1)

        o = tf.concat((Embedding(1)(tf.zeros((tf.shape(inputs)[0], 1))),  # CLS
                       tf.stack((Dense(d_model)(x_1),                     # PClass
                                 Dense(d_model)(x_3),                     # Age
                                 Dense(d_model)(x_4),                     # SibSp
                                 Dense(d_model)(x_5),                     # Parch
                                 Dense(d_model)(x_6)),                    # Fare
                                axis=1),
                       Embedding(2)(x_2),                                 # Sex。マジック・ナンバーが入ってしまってごめんなさい……
                       Embedding(4)(x_7),                                 # Embarked。マジック・ナンバーが入ってしまってごめんなさい……
                       Embedding(5)(x_8)),                                # Title。マジック・ナンバーが入ってしまってごめんなさい……
                      axis=1)

        for _ in range(block_size):
            o = Add()((o, SelfAttention()(LayerNormalization()(o))))
            o = Add()((o, FeedForwardNetwork()(LayerNormalization()(o))))

        o = Sigmoid()(Dense(1)(o[:, 0]))

        return o

    return op
