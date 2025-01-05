from django.urls import path
from SignLanguageDetection import Consumer


websocket_urlpatterns = [
    path('ws/sign/', Consumer.SignDec.as_asgi()),
]