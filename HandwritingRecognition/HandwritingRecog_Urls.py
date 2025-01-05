from django.urls import path, include
from HandwritingRecognition import views

urlpatterns = [
    path('',views.home,name='home'),

]