import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

dataFrame = pd.read_excel("maliciousornot.xlsx")
print(dataFrame.info())
print(dataFrame.describe())  # datamızı inceledik
print(dataFrame.corr())  # url uzunluğu arttıkça virus olma ihtimali artıyormuş mesela

sbn.countplot(x="Type", data=dataFrame)  # kaç tanesi temiz kaç tanesi virüslü
# plt.show()

dataFrame.corr()["Type"].sort_values().plot(kind="bar")  # coralasyon grafiği
# plt.show()

y = dataFrame["Type"].values
x = dataFrame.drop("Type", axis=1).values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print(x_train.shape)  # (383, 30)

model = Sequential()

model.add(Dense(units=30), activation="relu")
# içeriye 30 tane nöron koy dedik çünkü kolon sayısı kadar giriş nöronu öneriliyormuş
