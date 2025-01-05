from django.urls import path
from FaceRecognition import Consumer


websocket_urlpatterns = [
    path('ws/face/', Consumer.FaceRecog.as_asgi()),
]