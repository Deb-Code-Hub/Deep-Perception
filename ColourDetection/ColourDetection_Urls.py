from django.urls import path, include
from ColourDetection import views

urlpatterns = [
    path('',views.home,name='home'),
    #path('video_feed',views.video_feed,name='video_feed'),

]