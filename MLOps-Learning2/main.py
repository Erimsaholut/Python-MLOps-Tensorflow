import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/satislar.csv")

aylar = df[["Aylar"]]
satislar = df[["Satislar"]]
satislar2 = df.iloc[:, 1:2]

# todo verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split  # verimizi bölme işlemlerine başlıyoruz

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.3, random_state=0)
# random state minecrafttaki seed mantığıyla aynı


"""
# todo verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)  # yine 0 la 1 arasında küçültttü
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)  # predict=tahmin

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.title("Aylara Göre Satis")
plt.xlabel("Aylar")
plt.ylabel("Satislar")
plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))
plt.show()
