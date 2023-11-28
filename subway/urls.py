from django.urls import path, include
from .views import *

urlpatterns = [
    path('analyze/', analyze, name='analyze'),
    # path('xgboost/', xgboost, name='xgboost'),
]