import pandas as pd
import sklearn
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import keras
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

class QuantileReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.quantiles = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include='number'):
            low_quantile = X[col].quantile(self.threshold)
            high_quantile = X[col].quantile(1 - self.threshold)
            self.quantiles[col] = (low_quantile, high_quantile)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X.select_dtypes(include='number'):
            low_quantile, high_quantile = self.quantiles[col]
            rare_mask = ((X[col] < low_quantile) | (X[col] > high_quantile))
            if rare_mask.any():
                rare_values = X_copy.loc[rare_mask, col]
                replace_value = np.mean([low_quantile, high_quantile])
                if rare_values.mean() > replace_value:
                    X_copy.loc[rare_mask, col] = high_quantile
                else:
                    X_copy.loc[rare_mask, col] = low_quantile
        return X_copy


df = pd.read_csv('dataset.csv')

df = df.drop(columns=['n'])
df = df.fillna(0.351837) # Заменяем средним значением
df = df.dropna(how="all") # Удаляем столбец, если в нем все данные NaN

print(df.isna().sum())

onehotencoder = OneHotEncoder()
x = df.drop(columns=['Culture'])
y = pd.DataFrame(onehotencoder.fit_transform(df[['Culture']]).toarray(), columns=onehotencoder.get_feature_names_out(['Culture']))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.87, random_state=42)

model = keras.Sequential(
    [
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Normalization(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(13),
        keras.layers.Dense(13, activation='softmax')
    ]
)

model.compile(keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=50)

y_pred = pd.DataFrame(model.predict(x_test))
y_pred.columns = y_test.columns
y_pred = onehotencoder.inverse_transform(y_pred)

print(classification_report(onehotencoder.inverse_transform(y_test), y_pred, target_names=y_test.columns))
accuracy_score(onehotencoder.inverse_transform(y_test), y_pred)

filename = 'fmodel.pkl'
pickle.dump(model, open(filename, 'wb'))

test_df = pd.read_csv('private_dataset.csv')
test_df = test_df.drop(columns=['Unnamed: 0'])
predicts = pd.DataFrame(model.predict(x_test))
pd.DataFrame(onehotencoder.inverse_transform(predicts)).to_csv('answers.csv')
