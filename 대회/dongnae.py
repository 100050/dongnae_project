import tensorflow as tf
import numpy as np
import datetime as dt
import cv2
import matplotlib.pyplot as plt
import glob
import pandas as pd

# 추가해야할 것: LED불 들어오게하기, 어느시간에 어디에 들렸는지 확인

# 모델 학습 함수 (이해하기 어려움, 걍 모델이 이렇게 생겼구나라고 생각하셈)
def model_fit(paths):
    독립 = np.array([plt.imread(paths[i]) for i in range(len(paths))])
    종속 = np.array([paths[i].split('\\')[-2] for i in range(len(paths))])

    독립 = 독립.reshape(64, 480, 640, 3)
    x_test = 독립[:10]
    종속 = pd.get_dummies(종속)
    y_test = 종속[:10]

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

    # 모델을 학습
    model.fit(독립, 종속, epochs=20)
    # 정확도 출력
    loss, acc = model.evaluate(x_test, y_test)
    print("정확도: " + str(acc)*100 + "%")
    for i in range(len(paths)):
        with open("time_{}.txt".format(paths[i].split('\\')[1]), "w") as e:
                        e.write()
    # model.save('my_model.h5')

# 이미지 처리하기 (화면에 나오는 이미지를 예측할 수 있도록 사이즈 변경)
def preprocessing(frame):
    #frame_fliped = cv2.flip(frame, 1)
    # 사이즈 조정 티쳐블 머신에서 사용한 이미지 사이즈로 변경해준다.
    size = (480, 640)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    
    # 이미지 정규화
    # astype : 속성
    # frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1

    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    # keras 모델에 공급할 올바른 모양의 배열 생성
    frame_reshaped = frame_resized.reshape((1, 480, 640, 3))
    return frame_reshaped

# 예측용 함수 (재조정된 이미지를 여기서 불러와 예측)
def predict(frame):
    prediction = model.predict(frame)
    return prediction

# cv2 카메라 제어 함수
def capture_read(paths):
    #카메라 제어
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()

        now = dt.datetime.now().strftime("%d_%H-%M-%S")
        key = cv2.waitKey(33)

        preprocessed = preprocessing(frame)
        prediction = predict(preprocessed)
        a = np.argmax(prediction[0]) # 예측한 값중 제일 확률이 높은 값 가져오기
        # print(a)
        # esc 누르면 카메라 off
        if key == 27:
            break
        # 분류
        for i in range(len(paths)):
            if a == i:
                frame = cv2.putText(frame, paths[i].split('\\')[1], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
                with open("{}.txt".format(paths[i].split('\\')[1]), "r") as time:
                    time.write(str(now))

        cv2.imshow("VideoFrame", frame)

    capture.release()

# 모델 위치
model_filename ='my_model.h5'

# 케라스 모델 가져오기
model = tf.keras.models.load_model(model_filename)

# 이미지 읽어서 데이터 준비하기
paths = glob.glob('pan/*/*.png')
paths = np.random.permutation(paths) 

# 읽기 시작
capture_read(paths)

cv2.destroyAllWindows()