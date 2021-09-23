import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

# 모델 위치
model_filename ='my_model.h5'

# 케라스 모델 가져오기
model = tf.keras.models.load_model(model_filename)

# 이미지 읽어서 데이터 준비하기
paths = glob.glob('pan/*/*.png')
paths = np.random.permutation(paths)

종속 = np.array([paths[i].split('\\')[-2] for i in range(len(paths))])

print(종속)