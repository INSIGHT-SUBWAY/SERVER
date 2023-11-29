import requests
import json
import pandas as pd
from datetime import datetime
from collections import deque
import joblib
import numpy as np
from .data import *

# [함수 1] (입력) 역 이름 → (출력) 역 코드

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

# [함수 2] (입력) 역 이름, 시간, 상행(1)/하행(2) + SEOUL_KEY → (출력) 현재 열차 코드(TRAIN_CODE) + 무슨 행인지(SUBWAYEND) + 열차 도착 시간(ARRIVETIME)

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
    
    result_train = {
        'TRAIN_CODE': None,
        'SUBWAYEND': None,
        'ARRIVETIME': None
    }
    
    # "row" 키에 해당하는 값에서 "ARRIVETIME"이 TIME보다 큰 빠른 열차 번호 찾기
    min_time = float('inf')
    for train in parsed_data["SearchSTNTimeTableByIDService"]["row"]:
        if train.get("ARRIVETIME") >= TIME:
            train_time = int(train.get("ARRIVETIME").replace(":", ""))
            if train_time < min_time:
                min_time = train_time
                result_train['TRAIN_CODE'] = train["TRAIN_NO"]
                result_train['SUBWAYEND'] = train["SUBWAYENAME"]
                result_train['ARRIVETIME'] = train["ARRIVETIME"]
                

    if result_train['TRAIN_CODE'] == None:
        print("해당 열차가 존재하지 않습니다.")
        return -1
    
    return result_train

# [함수 3] (입력) 열차 번호 + SK_KEY -> (출력) 실시간 혼잡도

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

# [함수 4] inout_tag 계산 함수 (내선: 1, 외선: 2, 역 이름 오류: -1)

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
        return 1
    else:
        return 2
    
# [함수 4] 경로 리스트 받아오기
def route_list(start_station, end_station, inout_tag):
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

    route = []

    # in 일 때
    if inout_tag == 1:
        cur = in_dq.popleft()
        while (cur != start_station):
            in_dq.append(cur)
            cur = in_dq.popleft()

        while (cur != end_station):
            in_dq.append(cur)
            cur = in_dq.popleft()
            route.append(cur)
            
    # out 일 때
    if inout_tag == 2:
        cur = out_dq.pop()
        while (cur != start_station):
            out_dq.appendleft(cur)
            cur = out_dq.pop()

        while (cur != end_station):
            out_dq.appendleft(cur)
            cur = out_dq.pop()
            route.append(cur)

    return route

# [함수 5] 경로 최소 혼잡도
def minMeanIndex(array):
    column_means = []
    for col in range(len(array[0])):
        col_sum = sum(row[col] for row in array)
        col_mean = col_sum / len(array)
        column_means.append(col_mean)

    min_index = column_means.index(min(column_means))
    return min_index + 1

# [함수 6] 경로 중 최소 혼잡도
def minValueIndex(array):
    min_value = min(min(row) for row in array)
    for row in array:
        if min_value in row:
            return row.index(min_value) + 1

# [함수 7] 예측 평균 혼잡도
def meanArray(array):
    column_means = []
    for col in range(len(array[0])):
        col_sum = sum(row[col] for row in array)
        col_mean = col_sum / len(array)
        column_means.append(col_mean)

    return column_means

def route_congestion(start_station, end_station, inout_tag):
    # 모델 불러오기
    loaded_model = joblib.load('xgboost_model.pkl')

    # # 나중에 아래 거를 이걸로 수정
    # input_df =  # 전처리한 데이터프레임 받아오는 함수 (수빈 언니가 만들어 줄 것임)
    
    # 시연 영상용 모델 불러오기
    # 208_220.csv: 왕십리-선릉
    if start_station == '왕십리' and end_station == '선릉': 
        input_df =  pd.read_csv("./208_220.csv", index_col=0)
    # 240_205.csv: 신촌-동대문역사문화공원
    elif start_station == '신촌' and end_station == '동대문역사문화공원' :
        input_df =  pd.read_csv("./240_205.csv", index_col=0)
    # 233_202.csv: 대림-을지로입구
    elif start_station == '대림' and end_station == '을지로입구':
        input_df =  pd.read_csv("./233_202.csv", index_col=0)
    # 나중에 여기까지 지울 것
        
    prediction_df = make_predictions(input_df, loaded_model)

    # 여기도 지울 것
    prediction_df = prediction_df.drop(prediction_df.index[0])
    # 여기까지
    
    congestion_columns = [col for col in prediction_df.columns if col.startswith('Congestion')]
    
    PRED_LIST = prediction_df[congestion_columns].values.tolist()

    # 경로 리스트 받아오기
    ROUTE_LIST = route_list(start_station, end_station, inout_tag)
    
    # 경로 최소 혼잡도 받아오기
    MIN_MEAN_INDEX = minMeanIndex(PRED_LIST)
    
    # 경로 중 최소 혼잡도 받아오기
    MIN_VALUE_INDEX = minValueIndex(PRED_LIST)

    # 칸별 경로 평균 혼잡도 예측 리스트 받아오기
    MEAN_ARRAY = meanArray(PRED_LIST)
    
    # return list 만들기
    congestion_list = []

    for i in range(len(ROUTE_LIST)):
        congestion_list.append({ROUTE_LIST[i] : PRED_LIST[i]})
        
    # dictionary 형태로 합치기
    data = {
        'PRED_CONGESTION' : congestion_list,
        'MIN_MEAN_INDEX' : int(MIN_MEAN_INDEX),
        'MIN_VALUE_INDEX' : int(MIN_VALUE_INDEX),
        'MEAN_ARRAY' : MEAN_ARRAY
    }
    
    return data
