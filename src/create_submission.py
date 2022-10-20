import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from dataset import get_test_data_frame, get_xs


# モデルを読み込みます。
model = tf.keras.models.load_model('titanic-tf-transformer')
with open('categorical_features.pickle', mode='rb') as f:
    categorical_features = pickle.load(f)
with open('scaler_feature_ranges.pickle', mode='rb') as f:
    scaler_feature_ranges = pickle.load(f)

# データを取得します。
data_frame = get_test_data_frame()
xs_1, xs_2 = get_xs(data_frame, categorical_features, scaler_feature_ranges)

# 提出量のCSVを作成します。
submission = pd.DataFrame({'PassengerId': data_frame['PassengerId'], 'Survived': (model.predict_on_batch((xs_1, xs_2))[:, 0] >= 0.5).astype(np.int32)})
submission.to_csv('submission.csv', index=False)
