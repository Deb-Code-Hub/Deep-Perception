from django.urls import path, include
from HomePage import views


"""path('FaceRecognition',views.FaceRecognition,name="FaceRecognition"),
    path('ColorDetection',views.ColorDetection,name="ColorDetection"),
    path('ObjectDetection',views.ObjectDetection,name="ObjectDetection"),
    path('SignLangDetection',views.SignLangDetection,name="SignLangDetection"),
    path('HandwritingDetection',views.HandwritingDetection,name="HandwritingDetection"),"""
urlpatterns = [
    path('',views.home,name='home'),

]


