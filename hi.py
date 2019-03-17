# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax


# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_ar = np.array([[[0, 1, 2],
                 [3, 4, 5],
                 [3, 4, 5]],

                [[0, 1, 3],
                 [3, 9, 5],
                 [3, 4, 1]]])
y_ar = np.array([3, 5])
#(x_train, y_train), (x_test, y_test) = np.load("/home/ubuntu/.keras/datasets/mnist.npz")
x_train = x_ar
x_test = x_ar

y_train = y_ar
y_test = y_ar

# 데이터셋 전처리
x_train = x_train.reshape(2, 9).astype('float32') / 255.0
x_test = x_test.reshape(2, 9).astype('float32') / 255.0

# 원핫인코딩 (one-hot encoding) 처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 훈련셋과 검증셋 분리
x_val = x_train[:1] # 훈련셋의 30%를 검증셋으로 사용
x_train = x_train[1:]
y_val = y_train[:1] # 훈련셋의 30%를 검증셋으로 사용
y_train = y_train[1:]

# 2. 모델 구성하기 28-64 3-24
model = Sequential()
model.add(Dense(units=64, input_dim=3*3, activation='relu'))
model.add(Dense(units=6, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))

# 5. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('')
print('loss_and_metrics : ' + str(loss_and_metrics))

# 6. 모델 사용하기
xhat_idx = np.random.choice(x_test.shape[0], 1)
xhat = x_test[xhat_idx]
yhat = model.predict_classes(xhat)

for i in range(1):
    print('True : ' + str(argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))