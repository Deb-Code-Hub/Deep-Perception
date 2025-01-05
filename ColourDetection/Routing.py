from django.urls import path
from ColourDetection import Consumer


websocket_urlpatterns = [
    path('ws/colours/', Consumer.ColourDec.as_asgi()),
]