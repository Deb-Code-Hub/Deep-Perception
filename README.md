# Deep Perception: A Computer Vision Website

This project is a real-time computer vision website developed using the Django framework, offering the following five main features:

- **Colour Detection using HSV**
- **Object Detection using YOLOv3**
- **Sign Language Detection using SSD**
- **Face Recognition using HOG + Linear SVM**
- **Handwriting Recognition using CNN**

## Features

### Colour Detection using HSV
This feature allows users to detect and identify different colors in realtime video feeds using the HSV color space.

### Object Detection using YOLOv3
Utilizes the YOLOv3 algorithm to detect various objects in realtime video feeds.

### Sign Language Detection using SSD
Implements a Single Shot Multibox Detector (SSD) to recognize and interpret sign language gestures in realtime video feeds.

### Face Recognition using HOG + Linear SVM
Employs Histogram of Oriented Gradients (HOG) and a Linear Support Vector Machine (SVM) for face detection and recognition in realtime video feeds.

### Handwriting Recognition using CNN
Uses Convolutional Neural Networks (CNN) to analyze and recognize handwritten text in realtime feeds.

## Installation

### Required App Downloads/Install
- **Python 3.9 IDLE**
- **Pycharm IDE**
- **Postgresql installer** (postgresql-14.2-1-windows-x64)
- **pgAdmin 4** (pgadmin4-6.5-x64)

### Venv Python Packages Downloads/Install

Create a virtual environment and install the following packages:

1. Clone the repository:
   ```sh
   git clone https://github.com/Deb-Code-Hub/Deep-Perception.git
   
2. Install dependencies:
   ```sh
   pip install -r requirements.txt

3. Configure your Database Settings:
   
   Edit your settings.py file in your Django project to configure the database settings for PostgreSQL.
   
   Update the DATABASES dictionary:
   
   ```sh
   DATABASES = {
       'default': {
           'ENGINE': 'django.db.backends.postgresql',
           'NAME': 'your_database_name',
           'USER': 'your_database_user',
           'PASSWORD': 'your_database_password',
           'HOST': 'localhost',  # or your database host
           'PORT': '5432',       # default PostgreSQL port
          }
   }

5. Create the Database:
   
   Make sure you have created the database in PostgreSQL.

   You can create a database using pgAdmin or by running the following SQL command:
   
   ````sh
   CREATE DATABASE your_database_name;

7. Run Migrations:

   Apply the migrations to your PostgreSQL database using the following Django management commands:
   
   ```sh
   python manage.py makemigrations
   python manage.py migrate

8. Run the development server:
   ```sh
   python manage.py runserver

## Usage

After starting the development server, you can access the website at http://localhost:8000 and explore the different computer vision features.
   
