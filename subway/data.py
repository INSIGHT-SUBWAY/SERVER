import requests
import json
import pandas as pd
from datetime import datetime
from collections import deque
import joblib
import numpy as np
import math
import ast

import time
from sklearn.preprocessing import OneHotEncoder

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

# DOW, time feature engineering
def time_features(dataframe):
    # 요일을 숫자로 매핑
    day_of_week_mapping = {'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4}
    dataframe['Day of Week'] = dataframe['Day of Week'].map(day_of_week_mapping)

    # 시간 정보를 이용하여 새로운 변수 생성
    dataframe['sin_time'] = np.sin(2 * np.pi * dataframe['Hour'] / 24)
    dataframe['cos_time'] = np.cos(2 * np.pi * dataframe['Hour'] / 24)

    # 분 정보를 이용하여 새로운 변수 생성 (사인 및 코사인 변환)
    dataframe['sin_minute'] = np.sin(2 * np.pi * dataframe['Minute'] / 60)
    dataframe['cos_minute'] = np.cos(2 * np.pi * dataframe['Minute'] / 60)

    return dataframe

# drop & one-hot encoding
def drop_columns_and_one_hot_encode(dataframe):
    # drop
    columns_to_drop = ['Previous Station Name', 'HourMinute']
    dataframe.drop(columns=columns_to_drop, inplace=True)

    # one-hot encoding
    #dataframe = pd.get_dummies(dataframe, columns=['Route Code'])

    return dataframe

# Fast Transfer, Fast Get Off

def process_fast_columns(df):
    # 컬럼의 값을 리스트로 변환
    df['Fast Transfer'] = df['Fast Transfer'].apply(ast.literal_eval)
    df['Fast Get Off'] = df['Fast Get Off'].apply(ast.literal_eval)

    # 리스트의 값을 int로 변환하고 NumPy 배열로 변환
    df['Fast Transfer'] = df['Fast Transfer'].apply(lambda x: np.array([int(val) for val in x]))
    df['Fast Get Off'] = df['Fast Get Off'].apply(lambda x: np.array([int(val) for val in x]))

    # 'Fast Transfer' 컬럼의 값을 펼쳐서 새로운 컬럼에 할당
    for i in range(10):
        new_column_name = f'Fast Transfer{i}'
        df[new_column_name] = df['Fast Transfer'].apply(lambda x: x[i] if (isinstance(x, np.ndarray) and i < len(x)) else np.nan)

    # 'Fast Get Off' 컬럼의 값을 펼쳐서 새로운 컬럼에 할당
    for i in range(10):
        new_column_name = f'Fast Get Off{i}'
        df[new_column_name] = df['Fast Get Off'].apply(lambda x: x[i] if (isinstance(x, np.ndarray) and i < len(x)) else np.nan)

    # 원래 컬럼은 삭제 (옵션)
    df = df.drop(['Fast Transfer', 'Fast Get Off'], axis=1)

    return df

def one_hot_encode_for_route_code(dataframe):
    
    dataframe['Route Code_0.0'] = 0
    dataframe['Route Code_1.0'] = 0
    dataframe['Route Code_2.0'] = 0
    dataframe['Route Code_3.0'] = 0
    dataframe['Route Code_4.0'] = 0
    dataframe['Route Code_5.0'] = 0
    
    total_row_num = len(dataframe)
    for i in range(total_row_num):
        if(dataframe['Route Code'][i] == 0):
            dataframe.loc[dataframe.index[i],'Route Code_0.0'] = 1
        if(dataframe['Route Code'][i] == 1):
            dataframe.loc[dataframe.index[i],'Route Code_1.0'] = 1
        if(dataframe['Route Code'][i] == 2):
            dataframe.loc[dataframe.index[i],'Route Code_2.0'] = 1
        if(dataframe['Route Code'][i] == 3):
            dataframe.loc[dataframe.index[i],'Route Code_3.0'] = 1
        if(dataframe['Route Code'][i] == 4):
            dataframe.loc[dataframe.index[i],'Route Code_4.0'] = 1
        if(dataframe['Route Code'][i] == 5):
            dataframe.loc[dataframe.index[i],'Route Code_5.0'] = 1
    dataframe.drop('Route Code', axis=1, inplace=True)

    return dataframe
    

#2) 내선/외선(상행/하행) 알아내기

# in = 내선순환(0): 왼쪽에서 빼고 오른쪽으로 붙임
# out = 외선순환(1): 오른쪽에서 빼고 왼쪽으로 붙임

def in_out_tag(start_station, end_station):
    
    # 2호선 역이 맞는지 확인
    station_list = [
        '시청', '을지로입구', '을지로3가', '을지로4가', '동대문역사문화공원', '신당', '상왕십리', '왕십리', '한양대',
        '뚝섬', '성수', '건대입구', '구의', '강변', '잠실나루', '잠실', '잠실새내', '종합운동장', '삼성', '선릉',
        '역삼', '강남', '교대', '서초', '방배', '사당', '낙성대', '서울대입구', '봉천', '신림', '신대방',
        '구로디지털단지', '대림', '신도림', '문래', '영등포구청', '당산', '합정', '홍대입구', '신촌', '이대', '아현',
        '충정로'
    ]

    if (start_station not in station_list) or (end_station not in station_list):
        return -1

    in_dq = deque([
        '시청', '을지로입구', '을지로3가', '을지로4가', '동대문역사문화공원', '신당', '상왕십리', '왕십리', '한양대',
        '뚝섬', '성수', '건대입구', '구의', '강변', '잠실나루', '잠실', '잠실새내', '종합운동장', '삼성', '선릉',
        '역삼', '강남', '교대', '서초', '방배', '사당', '낙성대', '서울대입구', '봉천', '신림', '신대방',
        '구로디지털단지', '대림', '신도림', '문래', '영등포구청', '당산', '합정', '홍대입구', '신촌', '이대', '아현',
        '충정로'
    ])

    out_dq = deque([
        '시청', '을지로입구', '을지로3가', '을지로4가', '동대문역사문화공원', '신당', '상왕십리', '왕십리', '한양대',
        '뚝섬', '성수', '건대입구', '구의', '강변', '잠실나루', '잠실', '잠실새내', '종합운동장', '삼성', '선릉',
        '역삼', '강남', '교대', '서초', '방배', '사당', '낙성대', '서울대입구', '봉천', '신림', '신대방',
        '구로디지털단지', '대림', '신도림', '문래', '영등포구청', '당산', '합정', '홍대입구', '신촌', '이대', '아현',
        '충정로'
    ])

    in_cnt = 0
    out_cnt = 0

    # in_cnt 계산

    cur = in_dq.popleft()
    while (cur != start_station):
        in_dq.append(cur)
        cur = in_dq.popleft()

    while (cur != end_station):
        in_dq.append(cur)
        cur = in_dq.popleft()
        in_cnt = in_cnt + 1

    # out_cnt 계산

    cur = out_dq.pop()
    while (cur != start_station):
        out_dq.appendleft(cur)
        cur = out_dq.pop()

    while (cur != end_station):
        out_dq.appendleft(cur)
        cur = out_dq.pop()
        out_cnt = out_cnt + 1

    if (in_cnt <= out_cnt):
        return 0
    else:
        return 1

def station_code_to_name(station_code):
    station_dict = {
        '209': '한양대', '234': '신도림', '242': '아현', '229': '봉천', '243': '충정로', 
        '244': '용답', '235': '문래', '217': '잠실새내', '230': '신림', '218': '종합운동장', 
        '219': '삼성', '231': '신대방', '236': '영등포구청', '248': '양천구청', '204': '을지로4가', 
        '210': '뚝섬', '245': '신답', '202': '을지로입구', '226': '사당', '203': '을지로3가', 
        '211': '성수', '212': '건대입구', '237': '당산', '249': '신정네거리', '213': '구의', 
        '238': '합정', '250': '용두', '232': '구로디지털단지', '239': '홍대입구', '240': '신촌', 
        '241': '이대', '246': '신설동', '247': '도림천', '200': '까치산', '201': '시청', 
        '205': '동대문역사문화공원', '206': '신당', '207': '상왕십리', '214': '강변', '215': '잠실나루', 
        '216': '잠실', '220': '선릉', '221': '역삼', '222': '강남', '223': '교대', '224': '서초', 
        '225': '방배', '227': '낙성대', '228': '서울대입구', '208': '왕십리', '233': '대림'
    }

    station_code = str(station_code)
    if(station_code[0]=='0'):
        station_code = station_code[1:]
        
    if station_code not in station_dict:
        print("해당 역이 존재하지 않습니다.")
        return -1
        
    return station_dict[station_code]
    
# Station name -> Station Code로 변환해주는 함수
def station_name_to_code(station_name): # 입력 형식: string (station_NM = "신촌") -> 출력 형식: string(return = '0240')
    station_dict = {
        '한양대': '0209', '신도림': '0234', '아현': '0242', '봉천': '0229', '충정로': '0243', 
        '용답': '0244', '문래': '0235', '잠실새내': '0217', '신림': '0230', '종합운동장': '0218', 
        '삼성': '0219', '신대방': '0231', '영등포구청': '0236', '양천구청': '0248', '을지로4가': '0204', 
        '뚝섬': '0210', '신답': '0245', '을지로입구': '0202', '사당': '0226', '을지로3가': '0203', 
        '성수': '0211', '건대입구': '0212', '당산': '0237', '신정네거리': '0249', '구의': '0213', 
        '합정': '0238', '용두': '0250', '구로디지털단지': '0232', '홍대입구': '0239', '신촌': '0240', 
        '이대': '0241', '신설동': '0246', '도림천': '0247', '까치산': '0200', '시청': '0201', 
        '동대문역사문화공원': '0205', '신당': '0206', '상왕십리': '0207', '강변': '0214', '잠실나루': '0215', 
        '잠실': '0216', '선릉': '0220', '역삼': '0221', '강남': '0222', '교대': '0223', '서초': '0224', 
        '방배': '0225', '낙성대': '0227', '서울대입구': '0228', '왕십리': '0208', '대림': '0233'
    }

    if station_name not in station_dict:
        print("해당 역이 존재하지 않습니다.")
        return -1
        
    return station_dict[station_name]

#3) 출발역과 도착역 사이 존재하는 역의 개수 세기
def num_of_between_station(start_station_code, dest_station_code):
    start_station_name = station_code_to_name(start_station_code)
    dest_station_name = station_code_to_name(dest_station_code)
    
    up_down_direction = in_out_tag(start_station_name, dest_station_name)
    if(up_down_direction == 0):
        direction = '상행'
    else:
        direction='하행'
    
    start_station_code = int(start_station_code)
    dest_station_code = int(dest_station_code)
    
    between_station_num = dest_station_code - start_station_code
    between_station_num = abs(between_station_num)-1 
    #print(between_station_num)

    if (between_station_num > (41-between_station_num)):
        between_station_num = (41-between_station_num)
    
    return [direction, between_station_num]

#4) previous_station 알아내기
def find_prev_station(start_station_code, dest_station_code):
    start_station_name = station_code_to_name(start_station_code)
    dest_station_name = station_code_to_name(dest_station_code)
    
    up_down_direction = in_out_tag(start_station_name, dest_station_name)
    if(up_down_direction == 0): #시계방향(내선)
        previous_tmp = int(start_station_code) -1
        previous_station_code = str(previous_tmp)
        if(previous_station_code == '200'):
            previous_station_code = '243'

    else:
        previous_tmp = int(start_station_code) +1
        previous_station_code = str(previous_tmp)
        if(previous_station_code == '244'):
            previous_station_code = '201'
            
    return previous_station_code


#서울시 열린 데이터 광장: 실시간 역별 도착 열차정보 호출하는 함수
def train_code(STATION_NAME, TIME, INOUT_TAG, SEOUL_KEY):
    
    STATION_CD = station_name_to_code(STATION_NAME)
    if (STATION_CD == -1): 
        return -1
    
    # 입력값 -> 수정 X
    SERVICE = "SearchSTNTimeTableByIDService"
    START_INDEX = 0 # 페이징 시작번호: 데이터 행 시작번호
    END_INDEX = 200 # 페이징 끝번호 : 데이터 행 끝번호
    WEEK_TAG = 1 # 평일:1, 토요일:2, 휴일/일요일:3
    TYPE = "json"
    
    # 호출 url
    url = f"http://openAPI.seoul.go.kr:8088/{SEOUL_KEY}/{TYPE}/{SERVICE}/{START_INDEX}/{END_INDEX}/{STATION_CD}/{WEEK_TAG}/{INOUT_TAG}"
    response = requests.get(url)
    
    # 시간에 따른 열차 번호 찾기(주어진 시간 직후의 열차 번호) 
    #  입력값: 시간
    
    # JSON 데이터를 딕셔너리로 파싱
    parsed_data = json.loads(response.text)

    # "row" 키에 해당하는 값에서 "ARRIVETIME"이 TIME보다 큰 빠른 열차 번호 찾기
    fast_train = None
    min_time = float('inf')
    for train in parsed_data["SearchSTNTimeTableByIDService"]["row"]:
        if train.get("ARRIVETIME") >= TIME:
            train_time = int(train.get("ARRIVETIME").replace(":", ""))
            if train_time < min_time:
                min_time = train_time
                fast_train = train["TRAIN_NO"]

    if fast_train == None:
        print("해당 열차가 존재하지 않습니다.")
        return -1
    
    return fast_train

#서울시 열린 데이터 광장: 실시간 역별 도착 열차정보, 출발역, 종착역 호출하는 함수
def train_code_modified(STATION_NAME, TIME, INOUT_TAG, SEOUL_KEY):

    STATION_CD = station_name_to_code(STATION_NAME)
    if (STATION_CD == -1):
        return -1

    # 입력값 -> 수정 X
    SERVICE = "SearchSTNTimeTableByIDService"
    START_INDEX = 0 # 페이징 시작번호: 데이터 행 시작번호
    END_INDEX = 200 # 페이징 끝번호 : 데이터 행 끝번호
    WEEK_TAG = 1 # 평일:1, 토요일:2, 휴일/일요일:3
    TYPE = "json"

    # 호출 url
    url = f"http://openAPI.seoul.go.kr:8088/{SEOUL_KEY}/{TYPE}/{SERVICE}/{START_INDEX}/{END_INDEX}/{STATION_CD}/{WEEK_TAG}/{INOUT_TAG}"
    response = requests.get(url)

    # 시간에 따른 열차 번호 찾기(주어진 시간 직후의 열차 번호)
    #  입력값: 시간

    # JSON 데이터를 딕셔너리로 파싱
    parsed_data = json.loads(response.text)

    # "row" 키에 해당하는 값에서 "ARRIVETIME"이 TIME보다 큰 빠른 열차 번호 찾기
    fast_train = None
    route_start = None
    route_end = None
    min_time = float('inf')
    for train in parsed_data["SearchSTNTimeTableByIDService"]["row"]:
        if train.get("ARRIVETIME") >= TIME:
            train_time = int(train.get("ARRIVETIME").replace(":", ""))
            if train_time < min_time:
                min_time = train_time
                fast_train = train["TRAIN_NO"]
                route_start = train["SUBWAYSNAME"]
                route_end = train["SUBWAYENAME"]
                

    if fast_train == None:
        print("해당 열차가 존재하지 않습니다.")
        return -1
    
    
    return fast_train, route_start, route_end


#SK open api: 해당 열차번호의 실시간 칸별 혼잡도 호출하는 함수
def train_number_to_current_congestion_list(TRAIN_NUMBER, SK_KEY):
    url = f"https://apis.openapi.sk.com/puzzle/subway/congestion/rltm/trains/2/{TRAIN_NUMBER}"

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "appkey": SK_KEY}
    
    response = requests.get(url, headers=headers)
    result_response = json.loads(response.text)

    if result_response["success"] is False:
        print("해당 열차에 대한 실시간 혼잡도 정보가 존재하지 않습니다.")
        return -1

    
    # JSON 문자열을 파이썬 객체로 변환
    parsed_data = json.loads(response.text)
    
    # "congestionCar" 값을 파이썬 리스트로 변환
    congestion_car_list = parsed_data['data']['congestionResult']['congestionCar'].split('|')
    
    return congestion_car_list


#train_code() 함수 호출한 다음, train_number_to_current_congestion_list() 함수 호출하는 함수
def current_train_congestion_list(STATION_NAME, INOUT_TAG, SEOUL_KEY, SK_KEY):
    
    # 현재 시간 받아오기
    CURRENT_TIME = datetime.now().strftime('%H:%M:%S')
    
    # 현재 시간에 <STATION_NAME>에 들어오는 <INOUT_TAG>행 열차 번호 받아오기
    TRAIN_NUMBER = train_code(STATION_NAME, CURRENT_TIME, INOUT_TAG, SEOUL_KEY)

    if TRAIN_NUMBER == -1:
        return -1

    # 그 열차에 대한 실시간 혼잡도 리스트 출력
    CONGESTION_LIST = train_number_to_current_congestion_list(TRAIN_NUMBER, SK_KEY)
    
    if CONGESTION_LIST == -1:
        return -1
        
    return CONGESTION_LIST

#사용자의 출발 역의 열차의 칸 별 혼잡도 정보 호출
def real_time_congestion_call(start_station_name, dest_station_name):
    
    #dest_station_code = station_name_to_code(dest_station_name)
    #start_station_code = station_name_to_code(start_station_name)
    
    STATION_NAME =  start_station_name#📌 역 이름(문자열 형식)
    
    #출발역과 도착역으로 상/하행 방향 알아내기
    up_down_direction = in_out_tag(start_station_name, dest_station_name)
    
    #📌 상행,내선: 1 / 하행,외선: 2
    if (up_down_direction == 0): INOUT_TAG = 1 #내선일 때는 INOUT_TAG=1
    else: INOUT_TAG = 2

    SEOUL_KEY = "6e4a46554a70707933335467424375" #📌 서울 열린 데이터 광장 인증키
    SK_KEY = "DWJ3uJdzMy5lKNdoDgNUa2fe7zUjC3zw3UlfZ5h1" #📌 SK open API 앱키


    CURRENT_TIME = datetime.now().strftime('%H:%M:%S') #현재 시간 받아오기
    
    #요일 계산하기
    x = datetime.today().weekday()
    days = ['MON', 'TUE', 'WED', 'THU', 'FRI']
    TODAY_WEEKDAY = days[x]

    CONGESTION_LIST = current_train_congestion_list(STATION_NAME, INOUT_TAG, SEOUL_KEY, SK_KEY)
    
    result_data = []
    
    if (CONGESTION_LIST == -1):
        print("데이터 수집 실패")
    else:
        Hour, Minute, Second = CURRENT_TIME.split(":")
        if (INOUT_TAG == 1): #내선일때
            data_new = {
                'Station Name': STATION_NAME,
                'Up/Down': 0,
                'Hour': Hour,
                'Minute': Minute,
                'Second':Second,
                'Congestion List': CONGESTION_LIST,
                'Day of Week': TODAY_WEEKDAY
            }
        else: #외선일때
            data_new = {
                'Station Name': STATION_NAME,
                'Up/Down': 1,
                'Hour': Hour,
                'Minute': Minute,
                'Second':Second,
                'Congestion List': CONGESTION_LIST,
            'Day of Week': TODAY_WEEKDAY
        }
        

    result_data.append(data_new)
    
    # 딕셔너리를 데이터프레임으로 변환
    realtime_df = pd.DataFrame(result_data)

    return realtime_df

def df_add_init(START_STATION_CODE, DEST_STATION_CODE): 
    
    #출발역과 도착역으로 상/하행 방향 알아내기
    return_list = num_of_between_station(START_STATION_CODE, DEST_STATION_CODE)
    up_down_direction = return_list[0]
    
    if(up_down_direction == "상행"):
        UP_DOWN_LINE = 0
    else:
        UP_DOWN_LINE = 1
        
    #출발역의 이전역 알아내기
    PREVIOUS_STATION_CODE = find_prev_station(START_STATION_CODE, DEST_STATION_CODE)
    
    #현재 출발역의 실시간 혼잡도 정보 불러오기
    START_STATION_NAME = station_code_to_name(START_STATION_CODE)
    DEST_STATION_NAME = station_code_to_name(DEST_STATION_CODE)
    present_station_info = real_time_congestion_call(START_STATION_NAME, DEST_STATION_NAME)

    
    #현재 요일 정보
    TODAYWEEKDAY = present_station_info['Day of Week'][0]
    HOUR = present_station_info['Hour'][0]
    MINUTE = present_station_info['Minute'][0]
    
    df_add = {
        'Station': [START_STATION_CODE],
        'Previous Station Code': [PREVIOUS_STATION_CODE],
        'Previous Station Name': [station_code_to_name(PREVIOUS_STATION_CODE)],
        'Up/Down Line': [UP_DOWN_LINE],
        'Day of Week': [TODAYWEEKDAY],
        'Hour': [HOUR],
        'Minute': [MINUTE],
        'Route Code': [0],
        'Car Index': [0],
        'Congestion': [0],
        'Number of Transfer': [0],
        'Fast Transfer': [[]],
        'Fast Get Off': [[]],
        #'06-07시간대 승차': [0.0],
        #'06-07시간대 하차': [0.0],
        #'07-08시간대 승차': [0.0],
        #'07-08시간대 하차': [0.0],
        #'08-09시간대 승차': [0.0],
        #'08-09시간대 하차': [0.0],
        #'09-10시간대 승차': [0.0],
        #'09-10시간대 하차': [0.0],
        #'10-11시간대 승차': [0.0],
        #'10-11시간대 하차': [0.0]
    }
    df_add = pd.DataFrame(df_add)
    
    #사이역 개수 세서 데이터프레임에 추가할 행 개수 체크
    result_list = num_of_between_station(START_STATION_CODE, DEST_STATION_CODE)
    if (result_list[1] < (41-result_list[1])):
        between_station_num = result_list[1]
    else:
        between_station_num = 41- result_list[1]
    
    
    # df는 복사하려는 데이터프레임, row_index는 복사하려는 행의 인덱스, n은 복사 횟수
    row_to_copy = df_add.loc[[0]]  # 대괄호를 두 번 사용하여 DataFrame 형태를 유지
    rows_copied = pd.concat([row_to_copy] * (between_station_num+1), ignore_index=True)
    
    # 필요하다면, 복사된 행들을 원본 데이터프레임에 다시 추가
    df_add = pd.concat([df_add, rows_copied], ignore_index=True)
    
    
    #'Station'열 갱신
    for i in range(1, between_station_num+2):
        if(up_down_direction=="하행"): #하행(외선) : 역 코드 1씩 감소
            if( (int(df_add.loc[df_add.index[i-1],['Station']])-1) == 200):
                df_add.loc[df_add.index[i], 'Station']= 243
            else:
                df_add.loc[df_add.index[i], 'Station']= int(df_add.loc[df_add.index[i-1],['Station']])-1
        else: #상행(내선) : 역 코드 1씩 증가
            if( (int(df_add.loc[df_add.index[i-1],['Station']])+1) == 244):
                df_add.loc[df_add.index[i], 'Station']= 201
            else:
                df_add.loc[df_add.index[i], 'Station']=int(df_add.loc[df_add.index[i-1],['Station']])+1
    
    
    #'Previous Station Code'열 갱신
    for i in range(1, between_station_num+2):
        df_add.loc[df_add.index[i], 'Previous Station Code'] = df_add.loc[df_add.index[i-1], 'Station'] 
        
    #'Previous Station Name'열 갱신
    for i in range(1, between_station_num+2):
        df_add.loc[df_add.index[i], 'Previous Station Name']= station_code_to_name(df_add.loc[df_add.index[i],'Previous Station Code'])
    
    CONGESTION_LIST = present_station_info['Congestion List'][0]
    
    df_add.drop(df_add.index[0], inplace=True)
    df_add.reset_index(drop=True, inplace=True)
    return df_add, CONGESTION_LIST, present_station_info

#'시간': 문자열을 정수로
def hour_to_int(df_add):
    df_add['Hour'] =df_add['Hour'].astype(int)
    return df_add

#'분' 단위 조정: 내림
def make_minute_floor(df_add):
    
    df_add['Minute']= df_add['Minute'].astype(int) #'Minute'열 정수로 처리
    total_row_num = len(df_add)
    
    for i in range(1,total_row_num):
        df_add.loc[df_add.index[i], 'Minute'] = df_add.loc[df_add.index[i-1], 'Minute'] +2 #역 1개 지날 때마다 2분씩 추가
    
    df_add['Minute']=(df_add['Minute'].astype(int))//10*10 #일의 자리에서 내림
    
    for i in range(total_row_num):
        if (df_add.loc[df_add.index[i], 'Minute'] >= 60):
            df_add.loc[df_add.index[i], 'Minute'] = (df_add.loc[df_add.index[i], 'Minute']) % 60
            df_add.loc[df_add.index[i], 'Hour'] = df_add.loc[df_add.index[i], 'Hour'] + ((df_add.loc[df_add.index[i], 'Minute']) // 60) +1
    return df_add

#INOUT_TAG 설정하기
def start_and_end_of_train(INOUT_TAG, START_STATION_CODE, present_station_info):
    # 출발역, 종착역 받기
    STATION_NAME = station_code_to_name(START_STATION_CODE)
    TIME = present_station_info['Hour'][0]+":" + present_station_info['Minute'][0]+":" + present_station_info['Second'][0]
    SEOUL_KEY = "6e4a46554a70707933335467424375"
    fast_train, route_start, route_end= train_code_modified(STATION_NAME, TIME, INOUT_TAG , SEOUL_KEY)
    
    return fast_train, route_start, route_end

# 특정 경로의 Route Code값 반환
def route_num_new(inout_tag, route_start, route_end):
    if inout_tag == 1 : # 상행: 노선 0~2
        # 노선 0: 성수→을지로입구, 성수→신도림, 성수→홍대입구
        if route_start == "성수" and route_end == "을지로입구":
            return 0
        elif route_start == "성수" and route_end == "신도림":
            return 0
        elif route_start == "성수" and route_end == "홍대입구":
            return 0
        # 노선 1: 서울대입구→성수, 성수→삼성, 신도림→성수, 성수→서울대입구
        elif route_start == "서울대입구" and route_end == "성수":
            return 1
        elif route_start == "성수" and route_end == "삼성":
            return 1
        elif route_start == "신도림" and route_end == "성수":
            return 1
        elif route_start == "성수" and route_end == "서울대입구":
            return 1
        # 노선2: 홍대입구→성수, 삼성→성수, 성수→성수, 을지로입구→성수
        elif route_start == "홍대입구" and route_end == "성수":
            return 2
        elif route_start == "삼성" and route_end == "성수":
            return 2
        elif route_start == "성수" and route_end == "성수":
            return 2
        elif route_start == "을지로입구" and route_end == "성수":
            return 2
            
    elif inout_tag == 2 : # 하행: 노선 3~6
        # 노선3: 서울대입구 → 성수, 홍대입구 → 성수, 신도림 → 성수
        if route_start == "서울대입구" and route_end == "성수":
            return 3
        elif route_start == "홍대입구" and route_end == "성수":
            return 3
        elif route_start == "신도림" and route_end == "성수":
            return 3
        # 노선4: 성수 → 을지로입구, 성수 → 성수, 성수 → 삼성, 성수 → 홍대입구, 삼성 → 성수, 을지로입구 → 성수
        elif route_start == "성수" and route_end == "을지로입구":
            return 4
        elif route_start == "성수" and route_end == "성수":
            return 4
        elif route_start == "성수" and route_end == "삼성":
            return 4
        elif route_start == "성수" and route_end == "홍대입구":
            return 4
        elif route_start == "삼성" and route_end == "성수":
            return 4
        elif route_start == "을지로입구" and route_end == "성수":
            return 4
        # 노선5: 성수 → 서울대입구, 성수 → 신도림
        elif route_start == "성수" and route_end == "서울대입구":
            return 5
        elif route_start == "성수" and route_end == "신도림":
            return 5
        # 노선6: 신도림 → 까치산
        elif route_start == "신도림" and route_end == "까치산":
            return 6
        
# 데이터프레임에 ROUTE CODE 저장하기
def store_route_code_to_df(df_add, INOUT_TAG, START_STATION_CODE, present_station_info):
    fast_train, route_start, route_end = start_and_end_of_train(INOUT_TAG, START_STATION_CODE, present_station_info)
    route_code = route_num_new(INOUT_TAG, route_start, route_end)
    df_add['Route Code'] = route_code
    
    return df_add

def num_of_transfer_column(df_add, final):
    #df_add와 final의 'Station'열 모두 str로 처리
    df_add['Station']=df_add['Station'].astype(str)
    final['Station']=final['Station'].astype(str)
    # 'Station'을 키로 하고 'Number of Transfer'를 값으로 하는 딕셔너리 생성
    num_of_transfer_map = final.set_index('Station')['Number of Transfer'].to_dict()
    
    # df_add 데이터프레임의 'Station' 열에 매핑 적용
    df_add['Number of Transfer'] = df_add['Station'].map(num_of_transfer_map)
    return df_add


def fast_transfer_column(df_add, final):
    # 'Station'을 키로 하고 'Fast Transfer'를 값으로 하는 딕셔너리 생성
    fast_transfer_map = final.set_index('Station')['Fast Transfer'].to_dict()
    # df_add 데이터프레임의 'Station' 열에 매핑 적용
    df_add['Fast Transfer'] = df_add['Station'].map(fast_transfer_map)
    return df_add

def fast_getoff_column(df_add, final):
    # 'Station'을 키로 하고 'Fast Get Off'를 값으로 하는 딕셔너리 생성
    fast_getoff_map = final.set_index('Station')['Fast Get Off'].to_dict()

    # df_add 데이터프레임의 'Station' 열에 매핑 적용
    df_add['Fast Get Off'] = df_add['Station'].map(fast_getoff_map)
    
    return df_add

def get_inout_column(df_add, final, up_down_direction):
    if(up_down_direction==0): #내선일때
        final_tmp = final[final['Up/Down Line']==0]
    else:
        final_tmp = final[final['Up/Down Line']==1]
        
    getinout_map_1 = final_tmp.set_index('Station')['06-07시간대 승차'].to_dict()
    df_add['06-07시간대 승차'] = df_add['Station'].map(getinout_map_1)
    
    getinout_map_2 = final_tmp.set_index('Station')['06-07시간대 하차'].to_dict()
    df_add['06-07시간대 하차'] = df_add['Station'].map(getinout_map_2)
    
    
    getinout_map_3 = final_tmp.set_index('Station')['07-08시간대 승차'].to_dict()
    df_add['07-08시간대 승차'] = df_add['Station'].map(getinout_map_3)
        
    getinout_map_4 = final_tmp.set_index('Station')['07-08시간대 하차'].to_dict()
    df_add['07-08시간대 하차'] = df_add['Station'].map(getinout_map_4)
    
    
    
    getinout_map_5 = final_tmp.set_index('Station')['08-09시간대 승차'].to_dict()
    df_add['08-09시간대 승차'] = df_add['Station'].map(getinout_map_5)
    
    getinout_map_6 = final_tmp.set_index('Station')['08-09시간대 하차'].to_dict()
    df_add['08-09시간대 하차'] = df_add['Station'].map(getinout_map_6)
    
    
    getinout_map_7 = final_tmp.set_index('Station')['09-10시간대 승차'].to_dict()
    df_add['09-10시간대 승차'] = df_add['Station'].map(getinout_map_7)
    
    getinout_map_8 = final_tmp.set_index('Station')['09-10시간대 하차'].to_dict()
    df_add['09-10시간대 하차'] = df_add['Station'].map(getinout_map_8)
    
    
    
    getinout_map_9 = final_tmp.set_index('Station')['10-11시간대 승차'].to_dict()
    df_add['10-11시간대 승차'] = df_add['Station'].map(getinout_map_9)
    
    getinout_map_10 = final_tmp.set_index('Station')['10-11시간대 하차'].to_dict()
    df_add['10-11시간대 하차'] = df_add['Station'].map(getinout_map_10)
    
    
    return df_add

def car_index_column(df_add):
    total_row_num = len(df_add)
    
    result_df_add = pd.DataFrame(columns=df_add.columns)

    for i in range(total_row_num):
        row_to_copy = df_add.loc[[i]]  # 단일 행 복사
        # row_to_copy를 9번 반복하여 단일 데이터프레임으로 만들기
        rows_copied = pd.concat([row_to_copy] * 9, ignore_index=True)
        result_df_add = pd.concat([result_df_add, row_to_copy, rows_copied], ignore_index=True)

        
        #df_add = pd.concat(df_add.loc[i*10] + rows_copied, ignore_index=True)

    # 'Car Index' 열 값 할당
    for i in range(10 * total_row_num):
        result_df_add.loc[i, 'Car Index'] = i % 10

    return result_df_add

def congestion_column(df_add, CONGESTION_LIST):
    total_row_num = len(df_add)
    for i in range(total_row_num):
        idx = i%10
        df_add.loc[df_add.index[i], 'Congestion'] = CONGESTION_LIST[idx]
    return df_add

def make_new_column(df_add, CONGESTION_LIST):
    # 1. station, dow, hour, minute, rounte code 당 칸 별 congestion 하나로 합치기

    # 'station'를 기준으로 그룹화하고 'Congestion' 값을 배열로 합치기
    df_add1 = df_add.groupby(['Station', 'Previous Station Code', 'Previous Station Name', 'Up/Down Line',
                                'Day of Week', 'Hour', 'Minute', 'Route Code', 'Number of Transfer',
        'Fast Transfer', 'Fast Get Off', '06-07시간대 승차', '06-07시간대 하차',
        '07-08시간대 승차', '07-08시간대 하차', '08-09시간대 승차', '08-09시간대 하차',
        '09-10시간대 승차', '09-10시간대 하차', '10-11시간대 승차', '10-11시간대 하차'], sort=False).agg({'Congestion': lambda x: x.tolist()}).reset_index()

    # 새로운 컬럼을 생성하고 배열의 각 요소를 해당 컬럼에 할당
    for i in range(10):
        df_add1[f'Congestion{i}'] = 0

    # 'Congestion' 컬럼 및 중간 단계 컬럼 제거
    df_add1 = df_add1.drop(columns=['Congestion'], axis=1)
    
    # 2. 승/하차 인원을 시간대에 맞게 남기기

    # '승차인원' 및 '하차인원' 컬럼 생성
    for index, row in df_add1.iterrows():
        hour = row['Hour']
        df_add1.loc[index, 'GetIn_cnt'] = row[f'{hour:02d}-{hour+1:02d}시간대 승차']
        df_add1.loc[index, 'GetOut_cnt'] = row[f'{hour:02d}-{hour+1:02d}시간대 하차']

    # 'Hour' 컬럼 및 중간 단계 컬럼 제거
    df_add2 = df_add1.drop(columns=[f'{hour:02d}-' + f'{hour+1:02d}시간대 승차' for hour in range(6, 11)] +
                            [f'{hour:02d}-' + f'{hour+1:02d}시간대 하차' for hour in range(6, 11)])
    
    # 'Hour'와 'Minute'를 합친 새로운 컬럼 'HourMinute' 만들기
    df_add2['HourMinute'] = df_add2['Hour'].astype(str).str.zfill(2) + df_add2['Minute'].astype(str).str.zfill(2)
    
    
    # 3. 이전 역 칸별 혼잡도 컬럼 추가
    # 새로운 컬럼 PrevCongestion0, PrevCongestion2, PrevCongestion3, ,,,, PrevCongestion9 만들기
    for i in range(10):
        df_add2[f'PrevCongestion{i}'] = 0
        
        
    #첫 번째 행만 'PrevCongestion'열 채우기
    for i in range(10):
        col_name= f'PrevCongestion{i}'
        df_add2.loc[0, col_name] = CONGESTION_LIST[i]

    
    # NaN 값 처리
    df_add2.fillna(value=np.nan, inplace=True)
    
    df_add3 = df_add2.copy()

    # 결측값 처리
    df_add3.fillna(0, inplace=True)

    # 함수 적용
    df_add3 = time_features(df_add3)
    df_add3 = drop_columns_and_one_hot_encode(df_add3)
    df_add3 = one_hot_encode_for_route_code(df_add3)
    df_add3 = process_fast_columns(df_add3)
    
    if(df_add3.loc[0, 'Previous Station Code'][0]=='0'):
        df_add3.loc[0, 'Previous Station Code'] = df_add3.loc[0, 'Previous Station Code'][1:]

    return df_add3

def call_final_df():
    # 모든 컬럼을 보여주도록 설정
    pd.set_option('display.max_columns', None)
    
    final = pd.read_csv('final(11.25).csv', index_col=0)
    col_list = ['Unnamed:0','Station', 'Previous Station Code',
        'Previous Station Name', 'Up/Down Line', 'Day of Week', 'Hour',
        'Minute', 'Route Code', 'Car Index', 'Congestion', 'Number of Transfer',
        'Fast Transfer', 'Fast Get Off', '06-07시간대 승차', '06-07시간대 하차', '07-08시간대 승차',
        '07-08시간대 하차', '08-09시간대 승차', '08-09시간대 하차', '09-10시간대 승차',
        '09-10시간대 하차', '10-11시간대 승차', '10-11시간대 하차']
    final.columns = col_list

    final['Station'] = final['Station'].astype(str)
    return final

def main_func(start_station_name, dest_station_name):
    final = call_final_df()
    
    #역 이름과 역 코드 매칭
    start_station_code = station_name_to_code(start_station_name)
    dest_station_code = station_name_to_code(dest_station_name)
    
    #새로 데이터프레임 생성 시작 - final형태로 만들기
    df_add, CONGESTION_LIST, present_station_info = df_add_init(start_station_code, dest_station_code)
    
    df_add['Hour'] = 8
    df_add['Minute'] = 0
    
    INOUT_TAG = (in_out_tag(start_station_name, dest_station_name))+1
    df_add = store_route_code_to_df(df_add, INOUT_TAG, start_station_code, present_station_info)
    df_add = num_of_transfer_column(df_add, final)
    df_add = fast_transfer_column(df_add, final)
    df_add = fast_getoff_column(df_add, final)
    
    up_down_direction = in_out_tag(start_station_name, dest_station_name)
    df_add = get_inout_column(df_add, final, up_down_direction)
    df_add = get_inout_column(df_add, final, up_down_direction)
    
    df_add = car_index_column(df_add)
    df_add = congestion_column(df_add, CONGESTION_LIST)
    
    #modeling input용으로 가공
    new_df_add = make_new_column(df_add, CONGESTION_LIST)
    
    return new_df_add
