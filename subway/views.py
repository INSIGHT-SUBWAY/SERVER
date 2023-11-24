from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .functions import *
import requests
import json
import pandas as pd
from datetime import datetime
from django.http import JsonResponse
import math

# 1. 현재 들어오는 열차의 실시간 혼잡도
    # request: 출발역, 도착역
    # response: ~행, 불쾌 지수, 열차 도착 시간, 실시간 혼잡도 리스트, 탑승 최소 혼잡도
@api_view(['GET'])
def analyze(request):
    # request
    start_station = request.query_params.get('start_station') # 출발역
    end_station = request.query_params.get('end_station') # 도착역

    # 함수에 넣을 변수들
    STATION_NAME = start_station
    INOUT_TAG = in_out_tag(start_station, end_station)
    if (INOUT_TAG == -1):
        return JsonResponse({"error" : "입력하신 역의 정보가 존재하지 않습니다."}, status=status.HTTP_200_OK)
    SEOUL_KEY = "6e4a46554a70707933335467424375"
    SK_KEY = "ZWIH3CaKXp3ivs1nrevX5abFzWs9bZQpct4kwz1i"

    # 현재 시간 받아오기
    CURRENT_TIME = datetime.now().strftime('%H:%M:%S')

    # 현재 들어오고 있는 열차 정보 받아오기
    CURRENT_TRAIN = train_code(STATION_NAME, CURRENT_TIME, INOUT_TAG, SEOUL_KEY)
    if (CURRENT_TRAIN == -1):
        return JsonResponse({"error" : "현재 들어오는 열차 정보가 없습니다."}, status=status.HTTP_200_OK)
    
    # 그 열차에 대한 실시간 혼잡도 리스트 출력
    CONGESTION_LIST = train_number_to_current_congestion_list(CURRENT_TRAIN['TRAIN_CODE'], SK_KEY)
    
    if CONGESTION_LIST == -1:
        return JsonResponse({"error" : "현재 들어오는 열차의 혼잡도 정보가 없습니다."}, status=status.HTTP_200_OK)
    
    # DISCOMFORT_LEVEL (불쾌 지수) 계산
    CONGESTION_SUM = 0
    for congestion in CONGESTION_LIST:
        CONGESTION_SUM += int(congestion)

    CONGESTION_AVG = CONGESTION_SUM / len(CONGESTION_LIST)
    
    DISCOMFORT_LEVEL = CONGESTION_AVG

    # CURRENT_MIN_CONGESTION (탑승 최소 혼잡도) 계산
    min = 500
    min_car = 500
    for i in range(len(CONGESTION_LIST)):
        if int(CONGESTION_LIST[i]) <= min:
            min = int(CONGESTION_LIST[i])
            min_car = i

        
    CURRENT_MIN_CONGESTION_CAR = min_car + 1

    # response: ~행, 불쾌 지수, 열차 도착 시간, 실시간 혼잡도 리스트
    
    # 변수들을 하나의 딕셔너리로 합치기
    data = {
        'SUBWAYEND': CURRENT_TRAIN['SUBWAYEND'],
        'DISCOMFORT_LEVEL': DISCOMFORT_LEVEL,
        'ARRIVETIME': CURRENT_TRAIN['ARRIVETIME'],
        'CONGESTION_LIST': CONGESTION_LIST,
        'CURRENT_MIN_CONGESTION_CAR': CURRENT_MIN_CONGESTION_CAR
    }  

    return JsonResponse(data, status=status.HTTP_200_OK)
