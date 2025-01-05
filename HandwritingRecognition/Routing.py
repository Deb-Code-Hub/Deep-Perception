from django.urls import path
from HandwritingRecognition import Consumer


websocket_urlpatterns = [
    path('ws/hand/', Consumer.HandWriting.as_asgi()),
]