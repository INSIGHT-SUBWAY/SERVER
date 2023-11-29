import requests
import json
import pandas as pd
from datetime import datetime
from collections import deque
import joblib
import numpy as np
from .data import *

# 전처리 완료된 데이터프레임, model을 input으로 넣으면 예측 완료된 데이터프레임 반환 
def make_predictions(df, model):
    target_list = ['Congestion0', 'Congestion1', 'Congestion2', 'Congestion3', 'Congestion4', 'Congestion5', 'Congestion6', 'Congestion7', 'Congestion8', 'Congestion9']
    
    prev_target_list = ['PrevCongestion0', 'PrevCongestion1', 'PrevCongestion2',
                        'PrevCongestion3', 'PrevCongestion4', 'PrevCongestion5',
                        'PrevCongestion6', 'PrevCongestion7', 'PrevCongestion8',
                        'PrevCongestion9']
    length = len(df)

    # 반복적으로 예측과 값을 할당하는 함수
    def predict_and_assign(row_idx):
        # 현재 행의 feature 추출 (target_list에 해당하지 않는 컬럼만 추출)
        current_features = df.iloc[row_idx, ~df.columns.isin(target_list)].values.reshape(1, -1)

        # 예측값 반환
        predictions = model.predict(current_features)

        # 예측값을 현재 행의 target에 할당
        df.loc[row_idx, target_list] = predictions.flatten()

        # 예측값을 현재 행의 Prev target에 할당
        df.loc[row_idx + 1, prev_target_list] = predictions.flatten()

    # 데이터프레임의 행 수만큼 반복
    for i in range(length):
        # 예측 및 할당 수행
        predict_and_assign(i)
    
    df = df.iloc[:-1]
    
    return df