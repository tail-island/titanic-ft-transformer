import numpy as np
import pandas as pd
import os.path as path

from functools import reduce
from funcy import count, repeat


# DataFrameを取得します。
def get_data_frame(filename):
    return add_features(fill_na(pd.read_csv(path.join('..', 'input', 'titanic', filename))))


# 訓練用DataFrameを取得します。
def get_train_data_frame():
    return get_data_frame('train.csv')


# テスト用DataFrameを取得します。
def get_test_data_frame():
    return get_data_frame('test.csv')


# 数値型の特徴量のNaNを平均値で埋めます。
def fill_na(data_frame):
    for feature in ('Pclass', 'Age', 'SibSp', 'Parch', 'Fare'):
        data_frame[feature] = data_frame[feature].fillna(data_frame[feature].mean())

    return data_frame


# 特徴量を追加します。
def add_features(data_frame):
    # 肩書追加用の補助関数。
    def add_title(title_series, name_series, id, titles):
        title_series[reduce(lambda acc, series: acc + series, map(lambda title: name_series.str.contains(title), titles))] = id

        return title_series

    # 肩書を追加します。
    data_frame['Title'] = reduce(lambda title_series, params: add_title(title_series, data_frame['Name'], *params),
                                 ((0, ('Mr.', 'Dr.', 'Rev.', 'Don.', 'Col.', 'Major.', 'Capt.')),
                                  (1, ('Master.',)),
                                  (2, ('Mrs.', 'Mme.', 'Ms.')),
                                  (3, ('Miss.',))),
                                 pd.Series(repeat(np.nan, len(data_frame['Name'])), dtype='object'))

    return data_frame


# カテゴリ型の特徴量を、どの数値に変換するかのdictを取得します。
def get_categorical_features(data_frame):
    return dict(map(lambda feature: (feature, dict(zip(data_frame[feature].factorize()[1], count(1)))), ('Sex', 'Embarked', 'Title')))  # NaNは0にして、カテゴリは1始まり。


# 数値型の特徴量の最小値と最大値を取得します。正規化のためです。
def get_scalar_feature_ranges(data_frame):
    return dict(map(lambda feature: (feature, (data_frame[feature].min(), data_frame[feature].max())), ('Pclass', 'Age', 'SibSp', 'Parch', 'Fare')))


# 入力データを取得します。
def get_xs(data_frame, categorical_features, feature_ranges):
    # カテゴリ型の特徴量を、数値に変換します。
    for feature, mapping in categorical_features.items():
        data_frame[feature] = data_frame[feature].map(mapping).fillna(0)

    # 数値型の特徴量を、正規化します。
    for feature, (min, max) in feature_ranges.items():
        data_frame[feature] = (data_frame[feature] - min) / (max - min)

    # SexはNaNがないので、1を引いて0と1にします。
    data_frame['Sex'] = data_frame['Sex'] - 1

    # 予測に使用するカラムだけを抽出します。
    return data_frame[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']].values


# 出力データを取得します。
def get_ys(data_frame):
    return data_frame['Survived'].values
