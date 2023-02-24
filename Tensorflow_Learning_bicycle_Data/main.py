from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sbn
import pandas as pd
import openpyxl

df = pd.read_excel('bisiklet_fiyatlari.xlsx')

# VERİYİ TEST VE TRAİN OLARAK İKİYE AYIRMAK

# y= wx + b bizim denklem y yani fiyata gitmek istiyoruz. (w= weight)
y = df["Fiyat"].values
x = df[["BisikletOzellik1", "BisikletOzellik2"]].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=15)
# Veri rastgele şekilde %30una bölündü. bu random state'i iki kişi aynı veriyle aynı seçerse aynı çıkıyor

print(x_train.shape)  # 670 tanesi train
print(x_test.shape)  # 330 tanesi test için ayrılmış
print(y_train.shape, y_test.shape)  # y de aynı bölünmüş

# scailing veriyi  sıfırla bir arasına getirme işlemi yapıcaz şimdi
scaler = MinMaxScaler()  # scaler classını çıkarttık

scaler.fit(x_train)  # scaleri ayarladık

x_train = scaler.transform(x_train)  # scale ettik burada da
x_test = scaler.transform(x_test)  # scale ettik burada da

model = Sequential()  # modelimizi çıkardık

model.add(Dense(4, activation="relu"))  # bunlar playground'daki hidden layerlara denk geliyor.
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))

model.add(Dense(1))  # Bu da çıktı katmanına denk geliyor çok karışık, sınıflandırmalı değilse 1 tane yeterli imiş

model.compile(optimizer="rmsprop", loss="mse")
# mse = mean squared error

model.fit(x_train, y_train, epochs=250)
# burada modelimizi eğitiyoruz datalarla

loss = model.history.history["loss"]
# biraz önceki trainingten sonraki lossun değerlerini veriyor

sbn.lineplot(x=range(len(loss)), y=loss)
plt.show()
# bu grafik loss un küçülme grafiği kayıp ne kadar azsa o kadar iyi

train_lose = model.evaluate(x_train, y_train, verbose=0)  # train verisindeki loss değerimiz
test_lose = model.evaluate(x_test, y_test, verbose=0)  # test verisindeki loss değerimiz
print(test_lose, train_lose)  # bunların birbirlerine yakın çıkması gerekiyor ki kayıplar yakın olsun

testTahminleri = model.predict(x_test)  # modelin test datalarına göre oluşturduğu tahminler
tahminDf = pd.DataFrame(y_test, columns=["Gerçek Y"])  # gerçek değerler

testTahminleri = pd.Series(testTahminleri.reshape(330, ))  # grafikleri birleştirmek için Seriese çevirdik tahminleri

tahminDf = pd.concat([tahminDf, testTahminleri], axis=1)  # birleşitirdik burada
tahminDf.columns = ["Gerçek Y", "Tahmin Y"]
print(tahminDf)

sbn.scatterplot(x="Gerçek Y", y="Tahmin Y", data=tahminDf)  # bu da gerçek ve tahminin grafiksel karşılaştırması
plt.show()

print(mean_absolute_error(tahminDf["Gerçek Y"], tahminDf["Tahmin Y"]))

# Yeni data ekleyip tahmini fiyat oluşturucaz şimdi
newBicycleAttirbutes = [[1760, 1758]]  # datamız
newBicycleAttirbutes = scaler.transform(newBicycleAttirbutes)  # model için uygun hale getirmek için scale ettik
price = model.predict(newBicycleAttirbutes)  # ortaya çıkan fiyat
print(price)

# model.save("bicycle_model.h5") # modelimizi kaydettik .h5
#tekrar kaydetmemek için yoruma aldım uzantısı
# yuklenenModel = load_model("bicyle_model.h5") # ile de modeli yükleyebiliyoruz.

