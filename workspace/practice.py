import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.models import Model
from keras.layers import Dense
import matplotlib.pyplot as plt # 학습된 데이터 시각화
 
# 목표로 찾을 값을 반환하는 함수
def test_system(x):
    # return 0.4*x+8.0 # 입력 x에 대해, 0.4x+8의 값을 반환.
    return 0.01*x*x+2.0 #=0.01x제곱 + 2.0
 
 # 구체적으로 "어떤" 모델을 만들지 정의하는 함수임. 이 코드에서는 매우 단순한 구조의 모델을 만듬
def get_model():
    tf.random.set_seed(1000) # 모델의 초기 가중치가 랜덤하게 뽑히는데,그것의 시드를 고정하는 거임.

    # 여기 아래 두줄은 기존의 선형 모델 결과를 도출하기 위해 썼던 코드임.
    # input = Input(shape=(1,), name="Input") # 입력(input) 층 정의. (1,)은 입력 데이터가 하나(1)의 특성을 가지는 "스칼라 값"이라는 의미. 한 마디로 한번에 하나의 값만 들어오게 했다는 의미임.
    # output = Dense(1, activation='linear', name="Output")(input) # 출력(output)층 정의. 여기도 하나의 값만(1), 선형(linear)으로 나오도록 정의.

    input = tf.keras.Input(shape=(1,), name="Input") # 위와 동일
    hidden = Dense(2, activation='tanh', name="Hidden")(input) # 여기 이 부분을 통해 곡률이 있는 그래프가 그려질 수 있는 것.
    output = Dense(1, activation='linear', name='Output')(hidden)

    model = tf.keras.Model(inputs=[input], outputs=[output]) # 위에서 정의한 input/output을 하나로 만드는 과정임.
    opt = keras.optimizers.Adam(learning_rate=0.0015) # adam이라는 학습 방식을 사용할 거임. 해당 학습방식은 비유하자면 오답노트 정리 방식의 학습 방식임. 
    # learning_rate는 "학습률"이라 하며, 모델이 정답을 틀렸을 때 얼마나 경로를 바꿀지 그 비율을 정의. 
    # 모델이 틀릴 때마다 어떤 방향으로 수정할지 결정하는? 그런 학습 방법
    model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae']) # loss값이 적을 수록 학습이 잘되었다는 의미.
    model.summary()
    return model
 
 # if __name__ == '__main__': -> 해당 구문은, 해당 파이썬 파일을 "직접 실행할 때만" 코드를 실행하라는 의미로, 해당 소스코드가 모듈로써 다른 파일에서 실행되는 경우 갑자기 실행되는 상황을 막아줌.
if __name__ == '__main__':
    x_datas = np.array(range(-50, 51, 10))
    y_datas = []
    for x in x_datas:
        y_datas.append(test_system(x))
    y_datas = np.array(y_datas)
 
    # plt.scatter(x_datas, y_datas)
    # plt.show()
    model = get_model()
    
    history = model.fit(x_datas, y_datas, epochs=20000, shuffle=True) 
    # epochs => 학습 반복 횟수
    # 1000/4000/8000번으로 늘려가면서 테스트했음.
    plt.plot(history.history['loss'], 'b', label='loss')
    plt.show()
 
    x_test = np.array(range(-45, 45, 10))
    result = model.predict(x_test)
 
    plt.scatter(x_datas, y_datas, c='blue', s=5)
    plt.scatter(x_test, result, c='red', s=5)
    plt.show()
 
    weights = model.get_weights()
    print("W : ", weights[0], "   b : ", weights[1])
 