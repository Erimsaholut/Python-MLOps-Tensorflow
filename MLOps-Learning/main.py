import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/veriler.csv")
height = df[["boy"]]
height_weight = df[["boy", "kilo"]]

# todo Veri yükleme

eksikveriler = pd.read_csv("data/eksikveriler.csv")
# print(eksikveriler.dropna())  # eksik veri olan satırları attı
# print("\n")
# print(eksikveriler.fillna(25))  # bu da eksik olan satırları 25le doldurdu

from sklearn.impute import SimpleImputer  # farklı bir yol öğrenicez şimdi

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # np.nanların ortalaması
yas = eksikveriler.iloc[:, 1:4].values  # iloc = intager location
imputer = imputer.fit(yas[:, 1:4])  # bu kolonları öğrenmesini söyledik ortalamalarını aldı
yas[:, 1:4] = imputer.transform(yas[:, 1:4])  # öğrendiği değerleri uyguladı
print(yas)  # böylelikle boş değerleri sütunun ortlaması ile doldurduk

# -------------Veriler---------
# Kategorik-------- ------Sayısal
# Nominal-----------------Oransal(Ratio)
# Ordinal-----------------Aralık(Interval)

# nominal araba markaları gibi ne sıralayabilirsin ne karşılaştırabilirsin
# ordinal sokaktaki kapı numaraları, sıralama yapılabilir ama karşılaştırmak anlam belirtmez

# todo encoder Kategorik (Nominal Ordinal) ----> Numeric
ulke = eksikveriler.iloc[:, 0:1].values  # sadece ülkeleri aldık sayısala dönüştürücez
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()  # label encoding = le
ulke[:, 0] = le.fit_transform(eksikveriler.iloc[:, 0])
print(ulke)  # şu anda bütün ülkeleri 0,1,2 diye sayısal değerleri dönüştürdü

ohe = preprocessing.OneHotEncoder()  # one hot encoding
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)  # üç ülkeyi(tr,fr,us) sırasılya onehot encodera dönüştürdü
# one hat encoder ne, amk ne işimize yarıyor diye googleden baktım. bu printteki gibi yazılmasına deniyormuş
# ve makine öğrenmesi uygulamaları için iyiymiş bu şekilde formatlamak

# todo numpy dizileri dataframe dönüşümü
sonuc = pd.DataFrame(ulke, index=range(22), columns=["tr", "fr", "us"])
print(sonuc)

sonuc2 = pd.DataFrame(data=yas, index=range(22), columns=["boy", "kilo", "yas"])
print(sonuc2)

cinsiyet = eksikveriler.iloc[:, -1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=["Cinsiyet"])
print(sonuc3)

#todo dataframe birleştirme işlemi
sonsonuc = pd.concat([sonuc, sonuc2], axis=1)  # axis = 0 ise alt alta ekliyor, axis 1 ya yana ekliyor
sonsonuc2 = pd.concat([sonsonuc, sonuc3], axis=1)
print(sonsonuc)  # bu şekilde değiştirilmiş verilerimizle yeni bir dataframe oluşturduk.

# todo verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split  # verimizi bölme işlemlerine başlıyoruz

x_train, x_test, y_train, y_test = train_test_split(sonsonuc, sonuc3, test_size=0.3, random_state=0)
# random state minecrafttaki seed mantığıyla aynı

#todo verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)  # yine 0 la 1 arasında küçültttü
X_test = sc.fit_transform(x_test)
