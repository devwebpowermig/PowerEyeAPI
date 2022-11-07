from django.contrib import admin
from django.urls import path, include

from .views import (
    ParametersListApiView,
    ParametersDetailApiView,
    VideoCamApiView
)

from pwreyev309.PowerEye_v03 import PowerEyeCamApiView

from pwreye_api import urls as pwereye_urls

urlpatterns = [
    path('api', ParametersListApiView.as_view()),
    path('api/<int:parameters_id>', ParametersDetailApiView.as_view()),
    path('', VideoCamApiView.as_view()),
    path('powereye', PowerEyeCamApiView.as_view())

]
