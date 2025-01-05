from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponse
# Create your views here.


def home(request):
    return render(request,"Login.html")

def next(request):#next=login
    """if request.method == 'POST':
        name= request.POST['name']
        password= request.POST['pass']
        if password == "123":
            return render(request,"HomePage.html",{'name':name})
        else:
            return render(request,"Login.html",{'error':'PLEASE ENTER THE CORRECT PASSWORD'})"""
    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass']

        user = authenticate(username=username, password=pass1)

        if user is not None:
            login(request, user)
            fname = user.first_name
            messages.success(request, "Logged In Successfully!!")
            #print("Logged In Sucessfully!!")
            return render(request, "HomePage.html", {"name": fname})
        else:
            messages.warning(request, "Invalid Credentials!!")
            return redirect('next')
    else:
        return render(request, "Login.html")

def Register(request):
    if request.method == 'POST':
        username = request.POST['username']
        fname = request.POST['fname']
        lname = request.POST['lname']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']

        if User.objects.filter(username=username):
            messages.warning(request, "Username already exist! Please try some other username.")
            return redirect('Register')

        if User.objects.filter(email=email).exists():
            messages.warning(request, "Email Already Registered!!")
            return redirect('Register')

        if len(username) > 20:
            messages.warning(request, "Username must be under 20 characters!!")
            return redirect('Register')

        if pass1 != pass2:
            messages.warning(request, "Passwords didn't matched!!")
            return redirect('Register')


        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name = fname
        myuser.last_name = lname

        myuser.save()
        messages.success(request, "Account Created Sucessfully!! Please Login")
        return redirect('next')
    else:
        return render(request,"Register.html")

#def Login(request):
    #return render(request,"Login.html")

def Logout(request):
    logout(request)
    messages.success(request, "Logged Out Successfully!!")
    return redirect('next')
    #return render(request,"Login.html")