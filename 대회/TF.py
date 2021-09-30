import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pandas as pd 

# 이미지 읽어서 데이터 준비하기
paths = glob.glob('사람/*/*.jpg')
paths = np.random.permutation(paths) 
독립 = np.array([plt.imread(paths[i]) for i in range(len(paths))])
a = (paths[i].split('\\')[-2] for i in range(len(paths)))
종속 = np.array([paths[i].split('\\')[-2] for i in range(len(paths))])

독립 = 독립.reshape(len(list(a)), 960, 720, 3)
x_test = 독립[60:]
종속 = pd.get_dummies(종속)
y_test = 종속[60:]

# 모델을 완성합니다.
X = tf.keras.layers.Input(shape=[960, 720, 3])

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

# 모델을 학습
model.fit(독립, 종속, epochs=10)
# 정확도 출력
loss, acc = model.evaluate(x_test, y_test)
print("정확도: " + str(acc*100) + "%")
model.save('my_model2.h5')
city = ["Dootcamp_Alpha", "Manufacturing", "Training_Center", "River_Town", "Abandoned_Resort", "Banyan_Grove"]
for j in range(6):
    for i in range(1):
        with open("classification\\{}\\{}.txt".format(city[j], paths[i].split('\\')[1]), "w") as e:
            e.write(str(1))

