from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd
import numpy as np

dataFrame = pd.read_excel("merc.xlsx")

# print(dataFrame.head())
# print(dataFrame.describe())

print(dataFrame.isnull().sum())  # veride eksik var mı ona baktık, yokmuş

# plt.figure(figsize=(7, 5))
# sbn.distplot(dataFrame["price"])
# plt.show()

print(dataFrame["year"].value_counts())  # hangi yıldan kaç tane var.
# bunları kendi projende de kullan aq

# a = dataFrame["year"] > 2010  # burada da modeli 2010dan yüksek arabaları yazdırdık
# print(dataFrame[a])

print(dataFrame.corr())  # rakamlar arasındaki corrdinationu gösteriyor imiş.

print(dataFrame.corr()["price"].sort_values())  # sadece fiyatla ilgili corrları göster ve sırala
# mesela en çok üretim yılı ve engine size fiyatı etkiliyormuş. güzel bi veri bu en çok düşüren de kilometresi

sbn.scatterplot(x="mileage", y="price", data=dataFrame)
# plt.show() # kilometre arttıkça fiyat düşüyor grafiği

print(dataFrame.sort_values("price", ascending=False).head(20))
# artan(ascending)= false dediğiğimiz için çoktan aza ilk 20 tanesini getirdi
print(dataFrame.sort_values("price", ascending=True).head(20))
# Azdan çoğa # 650 paunda 2003 model mercedes # bizde bi 500 bini var

filteredDataFrame = dataFrame.sort_values("price", ascending=False).iloc[131:]
# en pahalı arabalardan, toplam datamızın %1 ine denk gelen 131 tanesini attık

plt.figure(figsize=(7, 5))
# sbn.distplot(filteredDataFrame["price"])
# plt.show() görüldüğü üzere daha dengeli oldu ve verinin %99u hala aynı

print(dataFrame.groupby("year").mean()["price"])
# dataframedaki dataları yıllara göre gruplayıp ortamasını alp fiyatlarını gösterdik

print(dataFrame[dataFrame.year != 1970].groupby("year").mean()["price"])
# lavuğun biri 70 model arabaya son model parası istemiş onu atmak için böyle yapabiliriz

dataFrame = filteredDataFrame
dataFrame = dataFrame[dataFrame.year != 1970]
# bu şekilde de %99luk veriyi asıl dataframe eşitleyip (bence gerek yoktu). 1970 olani attik

print(dataFrame.head())  # hocam transmission var ama sayısal bir değer olmadığı için hata verdirtir.

dataFrame = dataFrame.drop("transmission", axis=1)

# \O/ \O/ \O/ \O/ \O/ \O/ \O/ MODEL OLUŞTURMAYA GEÇİYORUZ \O/ \O/ \O/ \O/ \O/ \O/ \O/ \O/ \O/ \O/

y = dataFrame["price"].values
x = dataFrame.drop("price", axis=1).values

# from sklearn.model_selection import train_test_split'i çıkardık bölücez datayı
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=31)
# böldük datamızı test ve train olarak %30a 70 oranıyle

print(len(x_train))
print(len(x_test))  # bölünmüş valla

# from sklearn.preprocessing import MinMaxScaler ı çıkarttık
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# x_test ve x_train verisini hazırladık

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense ' bunları da çıkardık kırmızı yana yana

model = Sequential()

model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))  # 4 adet 12 nöronlu gizli katman ekledik

model.add(Dense(1))  # bu da çıkış katmanı activationa gerek yok çıkışta

model.compile(optimizer="adam", loss="mse")
# geçen sefer başka bir optimizer kullanmıştık bu sefer en performanslı olan adam ı kullandık

model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=250, epochs=300)
# validation_data ekledik bu sayede bi eğitilirken diğer datalara da bakaraktan eğitiliyor
# 13 bin datayı aynı anda vermemek için batch_size = 250 dedik bu değer çok önemli değil ama düşük olursa uzun sürermiş

lossData = pd.DataFrame(model.history.history)
print(lossData.head())
# kayıp datayı görüyoruz burada

lossData.plot()
plt.show()

# kayıp data grafiği buradaki de

# from sklearn.metrics import mean_squared_error,mean_absolute_error

tahmin_dizisi = model.predict(x_test)

print(mean_absolute_error(y_test, tahmin_dizisi))
# modelin tahminleri ve gerçek sonuçlara arasında 3000 poundluk bir fark var arabaların ortalaması 24bin pound %13 gibi
# sapmayı azaltmak için: Veriyi daha fazla temizleyebiliriz,Test_Size Split_Size değiştirebiliriz,epochsu arttırabiliriz
# noron ve katman sayımızı değiştirebiliriz. Çok abartırsak kendi verilerine göre oluşup başka veri gelince saçmayabilir

plt.scatter(y_test, tahmin_dizisi)
plt.plot(y_test, y_test, "g-*")
plt.show()

print(dataFrame.iloc[2])  # öylesine iloc[2]deki datayı çektik
newCarSeries = dataFrame.drop("price", axis=1).iloc[2]  # datamızıdan üsttekini çıkartarak yeni bir data oluşturduk
newCarSeries = scaler.transform(newCarSeries.values.reshape(-1, 5))  # scalerda scale ettik modele uygun olması için

print(model.predict(newCarSeries))  # bu da modelin tahminleri benim ilk denemede 65 binliği 60 bin tahmin etti
