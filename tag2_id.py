import cv2
import os

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return

save_all_frames('tag2data/Mal/Mal.mp4', 'tag2data/Mal', 'sample_Mal', 'jpg')
save_all_frames('tag2data/Al/Al(OH)2.mp4', 'tag2data/Al', 'sample_Al', 'jpg')
save_all_frames('tag2data/Lac/Lac.mp4', 'tag2data/Lac', 'sample_Lac', 'jpg')
save_all_frames('tag2data/MalAl/MalAl(OH)2.mp4', 'tag2data/MalAl', 'sample_MalAl', 'jpg')
save_all_frames('tag2data/MalLac/MalLac.mp4', 'tag2data/MalLac', 'sample_MalLac', 'jpg')
save_all_frames('tag2data/AlLac/Al(OH)2Lac.mp4', 'tag2data/AlLac', 'sample_AlLac', 'jpg')
save_all_frames('tag2data/None/None.mp4', 'tag2data/None', 'sample_None', 'jpg')
save_all_frames('tag2data/test/test.mp4', 'tag2data/test', 'test', 'jpg')
from PIL import Image
import os, glob
import numpy as np
from PIL import ImageFile
# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = ["Mal", "Al", "Lac", "MalAl","MalLac","AlLac","None"]
num_classes = len(classes)
image_size = 64
num_alldata = 100
num_testdata = 20


X_train = []
X_test  = []
y_train = []
y_test  = []

for index, classlabel in enumerate(classes):
    photos_dir = "./tag2data/" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        if i < num_testdata:
            X_test.append(data)
            y_test.append(index)
        elif i< num_alldata:
                X_train.append(data)
                y_train.append(index)

X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)

xy = (X_train, X_test, y_train, y_test)
np.save('./tag2data/Mal_Al_Lac.npy', xy)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from keras.utils import np_utils
import keras
import numpy as np

classes = ["Mal", "Al", "Lac","MalAl","MalLac","AlLac", "None"]
num_classes = len(classes)
image_size = 64

"""
データを読み込む関数
"""
def load_data():
    X_train, X_test, y_train, y_test = np.load("./tag2data/Mal_Al_Lac.npy", allow_pickle=True)
    # 入力データの各画素値を0-1の範囲で正規化(学習コストを下げるため)
    X_train = X_train.astype("float") / 255
    X_test  = X_test.astype("float") / 255
    # to_categorical()にてラベルをone hot vector化
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test  = np_utils.to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test

"""
モデルを学習する関数
"""
def train(X, y, X_test, y_test):
    model = Sequential()

    # X.shape[1:]とすることで、(64, 64, 3)となり、入力にすることが可能です。
    model.add(Conv2D(32,(3,3), padding='same',input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    # https://keras.io/ja/optimizers/
    # 今回は、最適化アルゴリズムにRMSpropを利用
    opt = RMSprop(lr=0.00005, decay=1e-6)
    # https://keras.io/ja/models/sequential/
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.fit(X, y, batch_size=28, epochs=10)
    # HDF5ファイルにKerasのモデルを保存
    model.save('./tag2data/cnn.h5')

    return model

"""
メイン関数
データの読み込みとモデルの学習を行います。
"""
def main():
    # データの読み込み
    X_train, y_train, X_test, y_test = load_data()
    X_train = np.array(X_train)
    # モデルの学習
    model = train(X_train, y_train, X_test, y_test)

main()

import keras
import sys, os
import numpy as np
from keras.models import load_model

imsize = (64, 64)

"""
dog1.jpgというファイル名の画像をGoogle Colab上にアップロードする方法は2通りあります。
1つが、下記のコードを実行し画像をアップロードする方法
from google.colab import files
uploaded = files.upload()
2つが、Colab左メニューの>アイコンを押して、目次、コード スニペット、ファイル
の3つ表示されるますが、右のファイルタブから画像をアップロードする方法です。
このファイルタブをクリックするとアップロードと更新の2つがありますが、
アップロードを押すと画像をアップロードすることが可能です。
"""
def load_image(path):
    img = Image.open(path)
    img = img.convert('RGB')
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    img = img.resize(imsize)
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0
    return img

import glob
import cv2

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

files_test = glob.glob("./tag2data/test/*.jpg")
files_test.sort()

keras_param = "./tag2data/cnn.h5"

x=[]
y1=[]
y2=[]
y3=[]

for n in range(len(files_test)):
    x.append(n)
    img = cv2.imread(files_test[n])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model = load_model(keras_param)
    img = load_image(files_test[n])
    prd = model.predict(np.array([img]))
    prelabel = np.argmax(prd, axis=1)
    if prelabel== 0:
      y1.append(1)
      y2.append(0)
      y3.append(0)
    elif prelabel == 1:
      y1.append(0)
      y2.append(2)
      y3.append(0)
    elif prelabel == 2:
      y1.append(0)
      y2.append(0)
      y3.append(3)
    elif prelabel == 3:
      y1.append(1)
      y2.append(2)
      y3.append(0)
    elif prelabel == 4:
      y1.append(1)
      y2.append(0)
      y3.append(3)
    elif prelabel == 5:
      y1.append(0)
      y2.append(2)
      y3.append(3)
    elif prelabel == 6:
      y1.append(0)
      y2.append(0)
      y3.append(0)




plt.figure(figsize=(12,8))
plt.plot(x,y1,color = 'red')
plt.plot(x,y2,color = 'blue')
plt.plot(x,y3,color = 'green')
plt.yticks([0,1,2,3],['None','Mal','Al(OH)2','Lac'])
plt.show()
