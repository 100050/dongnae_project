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
    city = ["Dootcamp_Alpha", "Manufacturing", "Training_Center", "River_Town", "Abandoned_Resort", "Banyan_Grove"]
    for j in range(6):
        for i in range(2):
            with open("classification\\{}\\{}.txt".format(city[j], paths[i].split('\\')[1]), "w") as e:
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

    city = ["Dootcamp_Alpha", "Manufacturing", "Training_Center", "River_Town", "Abandoned_Resort", "Banyan_Grove"]
    citys = [DA_button, Mt_button, TC_button, RT_button, AR_button, BG_button]

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
                citys[j].configure(fg="red")
                # led on

                frame = cv2.putText(frame, paths[i].split('\\')[1], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
                with open("classification/{}/{}.txt".format(city[j] ,paths[i].split('\\')[1]), "a") as time:
                    time.write(str(now) + "\n")

        cv2.imshow("VideoFrame", frame)

    citys[j].configure(fg="black")
    # led off   

    capture.release()

    cv2.destroyAllWindows()

# 검색 함수
def look_up():
    paths = glob.glob('pan/*/*.png')
    paths = np.random.permutation(paths) 
    city = ["Dootcamp_Alpha", "Manufacturing", "Training_Center", "River_Town", "Abandoned_Resort", "Banyan_Grove"]
    citys = [DA_button, Mt_button, TC_button, RT_button, AR_button, BG_button]
    for j in range(6):
        for i in range(2):
            with open("classification/{}/{}.txt".format(city[j] ,paths[i].split('\\')[1]), "r", encoding="UTF8") as time:
                a = time.readlines()
        if entry.get() == paths[i].split('\\')[1]:
            try:
                result.insert(tk.END, "{} 가 {} (으)로 {} 에 방문하였습니다.\n".format(entry.get(), city[j], a[-1]))
                citys[j].configure(fg="red")
                # led on
            except IndexError:
                result.insert(tk.END, "{} 가 {} (으)로 방문한 적이 없습니다.\n".format(entry.get(), city[j]))
                citys[j].configure(fg="black")
                # led off
# 초기화 함수
def resets():
    result.delete(1.0, tk.END)
    citys = [DA_button, Mt_button, TC_button, RT_button, AR_button, BG_button]
    for j in range(6):        
        citys[j].configure(fg="black")
        # led off
# 타이틀과 크기 설정
root = tk.Tk()
root.title("미아를 찾는 가장 빠른 방법")
root.geometry("1000x540+550+200") # 가로 * 세로, + x + y 좌표
root.resizable(False, False) #창 크기 변경 불가

# 머신러닝 버튼
menu = tk.Frame(root)
menu.pack(side="left", fill="both")

model = tk.Button(menu, width=16, height=100, text="머신러닝하기", command= lambda: model_fit())
model.pack()

# 지도
menu2 = tk.Frame(root, width=700, height=550)
menu2.pack(side="left")
canvas = tk.Canvas(menu2, width=600, height=450)
# cctv 버튼
photo = tk.PhotoImage(file="cctv1.png")
# 1번째
Dootcamp_Alpha = tk.Frame(canvas)
canvas.create_window((25, 150), window=Dootcamp_Alpha, anchor='nw')
DA_button = tk.Button(Dootcamp_Alpha, width=90, height=90, text="Dootcamp Alpha", font="Verdana 7",
 image=photo, compound=BOTTOM, command= lambda: capture_read(0))
DA_button.pack()
# 2번째
Manufacturing = tk.Frame(canvas)
canvas.create_window((150,25), window=Manufacturing, anchor='nw')
Mt_button = tk.Button(Manufacturing, width=90, height=90, text="Manufacturing", font="Verdana 7",
 image=photo, compound=BOTTOM, command= lambda: capture_read(1))
Mt_button.pack()
# 3번째
Training_Center = tk.Frame(canvas)
canvas.create_window((250,200), window=Training_Center, anchor='nw')
TC_button = tk.Button(Training_Center, width=90, height=90, text="Training Center", font="Verdana 7",
 image=photo, compound=BOTTOM, command= lambda: capture_read(2))
TC_button.pack()
# 4번째
River_Town = tk.Frame(canvas)
canvas.create_window((250,350), window=River_Town, anchor='nw')
RT_button = tk.Button(River_Town, width=90, height=90, text="River Town", font="Verdana 7",
 image=photo, compound=BOTTOM, command= lambda: capture_read(3))
RT_button.pack()
# 5번째
Abandoned_Resort = tk.Frame(canvas)
canvas.create_window((375,75), window=Abandoned_Resort, anchor='nw')
AR_button = tk.Button(Abandoned_Resort, width=90, height=90, text="Abandoned Resort", font="Verdana 7",
 image=photo, compound=BOTTOM, command= lambda: capture_read(4))
AR_button.pack()
# 6번째
Banyan_Grove = tk.Frame(canvas)
canvas.create_window((475,150), window=Banyan_Grove, anchor='nw')
BG_button = tk.Button(Banyan_Grove, width=90, height=90, text="Banyan Grove", font="Verdana 7",
 image=photo, compound=BOTTOM, command= lambda: capture_read(5))
BG_button.pack()

# 배경
background = tk.PhotoImage(file = "새비지.png")
canvas.create_image(0, 0, anchor = "nw", image = background)

canvas.pack(side="top")

# 검색
menu3 = tk.Frame(menu2, relief="solid")
menu3.pack(side="left", expand=True)
entry = tk.Entry(menu3, width=75)
entry.pack(side="left")

look_ups = tk.Button(menu3, text="검색", command=look_up)
look_ups.pack(side="right")


# 검색 결과
menu4 = tk.Frame(root, height=550)
menu4.pack(side="right", fill="both")

# 초기화 
head = tk.Frame(menu4)
head.pack(side="top")

reset = tk.Button(head, text="초기화", command=resets)
reset.pack(side="right")

lf = tk.LabelFrame(menu4, text="검색 결과", height=550)
lf.pack()
result = tk.Text(lf, height=550)
result.pack(fill="both")

root.mainloop()