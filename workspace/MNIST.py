import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Model
from keras.layers import Dense

all_files= [] # 이미지 파일 담을 리스트

for i in range(0, 10):
   path_dir = './images/training/{0}'.format(i)
   file_list = os.listdir(path_dir)
   file_list.sort()
   all_files.append(file_list) #대충 포맷대로 이미지를 리스트에 정렬해서 집어넣는다.

# x, y 트레이닝 데이터
x_train_datas = []
y_train_datas = []

for num in range(0, 10):
   for numbers in all_files[num]:
      img_path = './images/training/{0}/{1}'.format(num, numbers)
      print("load : " + img_path)
      img = Image.open(img_path)
      imgarr = np.array(img) / 255.0
      x_train_datas.append(np.reshape(imgarr, newshape=(784,1)))
      y_tmp = np.zeros(shape=(10))
      y_tmp[num] = 1
      y_train_datas.append(y_tmp)

eval_files= []
for i in range(0, 10):
   path_dir = './images/testing/{0}'.format(i)
   file_list = os.listdir(path_dir)
   file_list.sort()
   eval_files.append(file_list)

x_test_datas = []
y_test_datas = []
for num in range(0, 10):
   for numbers in eval_files[num]:
      img_path = './images/testing/{0}/{1}'.format(num, numbers)
      print("load : " + img_path)
      img = Image.open(img_path)
      imgarr = np.array(img) / 255.0 # 하나의 픽셀?마다 가능한 입력값이 255까지라서 255로 나누어줘야해.
      x_test_datas.append(np.reshape(imgarr, newshape=(784,1))) # 형태를 28X28=784개의 입력, 1의 출력으로 변형.
      y_tmp = np.zeros(shape=(10))
      y_tmp[num] = 1
      y_test_datas.append(y_tmp)

x_train_datas = np.reshape(x_train_datas, newshape=(-1, 784))
y_train_datas = np.reshape(y_train_datas, newshape=(-1, 10))
x_test_datas = np.reshape(x_test_datas, newshape=(-1, 784))
y_test_datas = np.reshape(y_test_datas, newshape=(-1, 10))

input = tf.keras.Input(shape=(784,), name="Input")
hidden = Dense(512, activation="relu", name="Hidden1")(input)
output = Dense(10, activation="softmax", name="Output")(hidden)

model = tf.keras.Model(inputs=[input], outputs=[output])
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
           optimizer=opt, metrics=['accuracy'])
model.summary()

history = model.fit(x_train_datas, y_train_datas, epochs=5, shuffle=True,
                    validation_data = (x_test_datas, y_test_datas))

plt.plot(history.history['loss'], 'b')
plt.plot(history.history['val_accuracy'], 'r')
plt.show()