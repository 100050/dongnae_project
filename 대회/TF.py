# 라이브러리 로딩
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
import cv2

# 이미지 읽어서 데이터 준비하기
paths = glob.glob('pan/*/*.png')
paths = np.random.permutation(paths)
독립 = np.array([plt.imread(paths[i]) for i in range(len(paths))])
종속 = np.array([paths[i].split('\\')[1] for i in range(len(paths))])
print(독립.shape, 종속.shape)

독립 = 독립.reshape(64, 480, 640, 3)
x_test = 독립[:10]
종속 = pd.get_dummies(종속)
y_test = 종속[:10]
print(독립.shape, 종속.shape)

# 모델을 완성합니다. 
X = tf.keras.layers.Input(shape=[480, 640, 3])

H = tf.keras.layers.Conv2D(9, kernel_size=(3, 3), activation='relu')(X)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Conv2D(18, kernel_size=(3, 3), activation='relu')(H)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Conv2D(36, kernel_size=(3, 3), activation='relu')(H)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Conv2D(72, kernel_size=(3, 3), activation='relu')(H)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='relu')(H)
H = tf.keras.layers.Dense(64, activation='relu')(H)
Y = tf.keras.layers.Dense(2, activation='softmax')(H)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# 모델을 학습합니다.
model.fit(독립, 종속, epochs=3)

# a = np.argmax(독립)
# print(a)
# 모델을 이용합니다. 
pred = model.predict(독립[0:5])
print(pd.DataFrame(pred).round(2))

loss, acc = model.evaluate(x_test, y_test)
print("정확도 : {:.2f}".format(acc))

# 종속변수를 가져오는 방법을 구해야함
# 정답 확인
# correct = tf.equal(tf.argmax(독립, 1), tf.argmax(종속, 1))
# acuu = tf.reduce_mean(tf.cast(correct, "float"))
# print(acuu) 
 
# 모델 확인
model.summary()

# 모델 저장
# model.save('my_model.h5')
