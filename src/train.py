import numpy as np
import tensorflow as tf

from dataset import get_categorical_features, get_scalar_feature_ranges, get_train_data_frame, get_xs, get_ys
from funcy import identity, juxt
from model import create_model
from parameter import ATTENTION_DROPOUT, BATCH_SIZE, BLOCK_SIZE, D_MODEL, EPOCH_SIZE, FFN_DROPOUT, FFN_FACTOR, HEAD_SIZE


rng = np.random.default_rng()

data_frame = get_train_data_frame()
categorical_features = get_categorical_features(data_frame)
scaler_feature_ranges = get_scalar_feature_ranges(data_frame)

xs = get_xs(data_frame, categorical_features, scaler_feature_ranges)
ys = get_ys(data_frame)

indices = rng.permutation(np.arange(len(xs)))

train_xs = xs[indices[200:]]
train_ys = ys[indices[200:]]

valid_xs = xs[indices[:200]]
valid_ys = ys[indices[:200]]

op = create_model(BLOCK_SIZE, D_MODEL, HEAD_SIZE, FFN_FACTOR, ATTENTION_DROPOUT, FFN_DROPOUT)

model = tf.keras.Model(*juxt(identity, op)((tf.keras.Input(shape=np.shape(xs)[1:]))))
model.summary()

model.compile(tf.keras.optimizers.experimental.AdamW(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=(tf.keras.metrics.BinaryAccuracy(),))
model.fit(train_xs, train_ys, BATCH_SIZE, EPOCH_SIZE, validation_data=(valid_xs, valid_ys))
