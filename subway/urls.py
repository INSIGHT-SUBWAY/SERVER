from django.urls import path, include
from .views import *

urlpatterns = [
    path('current_congestion_list/', current_congestion_list, name='current_congestion_list'),
]