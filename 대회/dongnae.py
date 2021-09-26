import tkinter as tk
from tkinter.constants import BOTTOM
import tensorflow as tf
import numpy as np
import datetime as dt
import cv2
import matplotlib.pyplot as plt
import glob
import pandas as pd 

#모델 학습 함수
def model_fit():
    # 이미지 읽어서 데이터 준비하기
    paths = glob.glob('pan/*/*.png')
    paths = np.random.permutation(paths) 
    독립 = np.array([plt.imread(paths[i]) for i in range(len(paths))])
    a = (paths[i].split('\\')[-2] for i in range(len(paths)))
    종속 = np.array([paths[i].split('\\')[-2] for i in range(len(paths))])

    독립 = 독립.reshape(len(list(a)), 480, 640, 3)
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
        with open("classification\\{}.txt".format(paths[i].split('\\')[1]), "w") as e:
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
    # 모델 위치
    model_filename ='my_model.h5'

    # 케라스 모델 가져오기
    model = tf.keras.models.load_model(model_filename)

    prediction = model.predict(frame)
    return prediction

# cv2 카메라 제어 함수
def capture_read(j):
    # 이미지 읽어서 데이터 준비하기
    paths = glob.glob('pan/*/*.png')
    paths = np.random.permutation(paths) 
    #카메라 제어
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    capture = cv2.VideoCapture(0)
    
    while True:
        ret, frame = capture.read()

        global now
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
        for i in range(2):
            if a == i:
                frame = cv2.putText(frame, paths[i].split('\\')[1], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
                with open("classification/{}/{}.txt".format(j ,paths[i].split('\\')[1]), "a") as time:
                    time.write(str(now) + "\n")

        cv2.imshow("VideoFrame", frame)

    capture.release()

    cv2.destroyAllWindows()

# 검색 함수
def look_up():
    paths = glob.glob('pan/*/*.png')
    paths = np.random.permutation(paths) 
    city = ["Dootcamp_Alpha", "Manufacturing", "Training_Center", "River_Town", "Abandoned_Resort", "Banyan_Grove"]
    for j in range(6):
        for i in range(2):
            with open("classification/{}/{}.txt".format(j+1 ,paths[i].split('\\')[1]), "r", encoding="UTF8") as time:
                a = time.readlines()
        if entry.get() == paths[i].split('\\')[1]:
            try:
                print("{} 가 cctv {} 에서 {} 에 방문하였습니다.".format(entry.get(), city[j], a[-1]))
                Manufacturing.configure(relief="solid", bd=3, highlightbackground="red")
            except IndexError:
                print("{} 가 cctv {} 에서 방문한 적이 없습니다.".format(entry.get(), city[j]))
                    

# 타이틀과 크기 설정
root = tk.Tk()
root.title("미아를 찾는 가장 빠른 방법")
root.geometry("720x540+550+200") # 가로 * 세로, + x + y 좌표
root.resizable(False, False) #창 크기 변경 불가

# 머신러닝 버튼
menu = tk.Frame(root, relief="solid", bd=1)
menu.pack(side="left", fill="both")

model = tk.Button(menu, width=15, height=100, text="머신러닝하기", command= lambda: model_fit())
model.pack()

# 지도
canvas = tk.Canvas(root, width = 600, height = 450)

# cctv 버튼
photo = tk.PhotoImage(file="cctv1.png")
# 1번째
Dootcamp_Alpha = tk.Frame(canvas, width=66, height=65)
canvas.create_window((25, 150), window=Dootcamp_Alpha, anchor='nw')
button = tk.Button(Dootcamp_Alpha, text="Dootcamp_Alpha", image=photo, compound=BOTTOM, command= lambda: capture_read(1)).pack()
# 2번째
Manufacturing = tk.Frame(canvas, width=66, height=65)
canvas.create_window((150,25), window=Manufacturing, anchor='nw')
button = tk.Button(Manufacturing, text="Manufacturing", image=photo, compound=BOTTOM, command= lambda: capture_read(2)).pack()
# 3번째
Training_Center = tk.Frame(canvas, width=66, height=65)
canvas.create_window((250,200), window=Training_Center, anchor='nw')
button = tk.Button(Training_Center, text="Training_Center", image=photo, compound=BOTTOM, command= lambda: capture_read(3)).pack()
# 4번째
River_Town = tk.Frame(canvas, width=66, height=65)
canvas.create_window((250,350), window=River_Town, anchor='nw')
button = tk.Button(River_Town, text="River_Town", image=photo, compound=BOTTOM, command= lambda: capture_read(4)).pack()
# 5번째
Abandoned_Resort = tk.Frame(canvas, width=66, height=65)
canvas.create_window((400,75), window=Abandoned_Resort, anchor='nw')
button = tk.Button(Abandoned_Resort, text="Abandoned_Resort", image=photo, compound=BOTTOM, command= lambda: capture_read(5)).pack()
# 6번째
Banyan_Grove = tk.Frame(canvas, width=66, height=65)
canvas.create_window((500,150), window=Banyan_Grove, anchor='nw')
button = tk.Button(Banyan_Grove, text="Banyan_Grove", image=photo, compound=BOTTOM, command= lambda: capture_read(6)).pack()

# 배경
background = tk.PhotoImage(file = "새비지.png")
canvas.create_image(0, 0, anchor = "nw", image = background)

canvas.pack()

# 검색
menu2 = tk.Frame(root, relief="solid", bd=1)
menu2.pack(side="right", expand=True)
entry = tk.Entry(menu2, width=75)
entry.pack(side="left")

look_up = tk.Button(menu2, text="검색", command=look_up)
look_up.pack(side="right")

root.mainloop()