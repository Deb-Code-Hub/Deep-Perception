"""DeepPerception URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('',include('Login.Login_Urls')),
    #path('next',include('HomePage.HomePage_Urls')),
    path('Go_To_Home',include('HomePage.HomePage_Urls')),
    #path('Logout',include('Login.Login_Urls')),
    path('ColorDetection',include('ColourDetection.ColourDetection_Urls')),
    path('ObjectDetection',include('ObjectDetection.ObjectDetection_Urls')),
    path('SignLangDetection', include('SignLanguageDetection.Sign_languages_URL')),
    path('FaceRecognition', include('FaceRecognition.Face_Recignition_Urls')),
    path('HandwritingDetection',include('HandwritingRecognition.HandwritingRecog_Urls')),
    path('admin/', admin.site.urls),
]
