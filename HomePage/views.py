from django.shortcuts import render
import cv2
from django.http import HttpResponse
# Create your views here.


def home(request):
    return render(request,"HomePage.html")
"""def FaceRecognition(request):
    return render(request, "FaceRecognition.html")
def ColorDetection(request):
    return render(request, "ColorDetection.html")
def ObjectDetection(request):
    return render(request, "ObjectDetection.html")
def SignLangDetection(request):
    return render(request, "SignLangDetection.html")
def HandwritingDetection(request):
    return render(request, "HandwritingDetection.html")"""

