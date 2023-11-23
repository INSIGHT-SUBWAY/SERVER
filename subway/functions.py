import requests
import json
import pandas as pd
from datetime import datetime

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

    # 여기에 추가적인 코드를 넣을 수 있습니다.
    
    # JSON 문자열을 파이썬 객체로 변환
    parsed_data = json.loads(response.text)
    
    # "congestionCar" 값을 파이썬 리스트로 변환
    congestion_car_list = parsed_data['data']['congestionResult']['congestionCar'].split('|')
    
    return congestion_car_list