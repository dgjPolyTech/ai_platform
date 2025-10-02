import tensorflow as tf
from tensorflow import keras
import numpy as np

import matplotlib.pyplot as plt

imdb = keras.datasets.imdb # keras의 imdb 영화 사이트 관련 dataset을 사용. 정확히는 리뷰에 주로 나오는 단어들의 묶음 같은 개념이다.
# 훈련시킬 데이터 묶음, 정답 데이터 묶음을 imdb 데이터를 토대로 구성한다.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#print("훈련 샘플: {},  레이블: {}".format(len(train_data), len(train_labels)))
# 리뷰 데이터의 경우, 데이터 샘플의 길이가 다 제각각이다(리뷰를 다 같은 길이로 적지 않을 테니까 당연함)
# 따라서 원활한 분석을 위해 데이터 전처리를 하는게 아래의 과정
# 우리는 최종적으로 단어를 정수형으로 변환하고, 문장에서 필요없는 부분이나 초과하는 부분은 자르도록 설계함.

# 단어와 정수 인덱스를 매핑한 딕셔너리
word_index = imdb.get_word_index()
# 처음 몇 개 인덱스는 사전에 정의되어 있음.
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) #One-Hot Encoding: 사전을 바탕으로 단어 하나를 대응되는 특정한 정수로 변환하는 것
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_review(train_data[0]))

#train_data를 빈칸은 0으로 앞에(POST) 채우고(<PAD>를 위에서 0으로 설정했으니까), 256을 최대 길이로 한다. 
# 굳이 256인 이유는 대충 DB와 연관되어 있음.
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
value=word_index["<PAD>"],
padding='post',
maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
value=word_index["<PAD>"],
padding='post',
maxlen=256)

# 이 코드는 결과적으로 긍정/부정 2가지 결과를 출력하는데, 시그모이드 함수를 써서 0쪽이면 부정, 1쪽이면 긍정으로 분류함.
# 워드 임베딩: 컴퓨터가 단어의 의미를 이해할 수 있게끔 단어를 숫자의 집합으로 바꾸는 것. > 사람은 "인간"이란 말을 들으면 인간=사람 같은 단어인걸 알지만, 컴퓨터는 모름.
# 그래서 워드 임베딩은 이런 유사한 의미의 단어를 비슷한 좌표끼리 묶어서, 적당히 알아먹을 수 잇게 해주는 것.

vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16,input_shape=(None,)))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(
partial_x_train,
partial_y_train,
epochs=40,
batch_size=512,
validation_data=(x_val, y_val),
verbose=1
)

results = model.evaluate(test_data, test_labels, verbose=2)
print(results)

history_dict = history.history
print(history_dict.keys())

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
# "bo"는 "파란색 점"입니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# b는 "파란 실선"입니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf() # 그림을 초기화합니다
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()