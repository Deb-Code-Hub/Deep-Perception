from django.urls import path
from ObjectDetection import Consumer


websocket_urlpatterns = [
    path('ws/object/', Consumer.ObjDec.as_asgi()),
]