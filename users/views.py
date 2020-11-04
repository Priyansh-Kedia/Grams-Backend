from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Profile
from django.utils import timezone
from .serializers import OTPSerializer
import random

@api_view(['POST','GET'])
def generate_otp(request):
    if request.method=="GET":
        print('TEST PASSED!!')
        return Response({'TEST':'test passed'})
    if request.method=="POST":
        user_profile=None
        phone_number = request.POST.get('phone_number')
        otp = random.randint(1111,9999)
        try:
            user_profile = Profile.objects.get(phone_number=phone_number)
        except Profile.DoesNotExist:
            user_profile = Profile.objects.create(phone_number=phone_number)
            

        if user_profile:
            user_profile.otp = otp
            user_profile.save()
            print (otp)
            data = {'success':'OTP Generated Successfully!','OTP': otp}
            return Response(data, status=status.HTTP_200_OK)
        else:
            return Response({'error':'User Does Not Exist'},status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST',])
def verify_otp(request):
    otp = request.POST.get('otp')
    phone_number = request.POST.get('phone_number')
    user_profile=None

    try:
        user_profile = Profile.objects.get(phone_number=phone_number)
    except Profile.DoesNotExist:
        pass

    if user_profile:
        if otp==str(user_profile.otp) and (timezone.now()-otp_timestamp).seconds < 30:
            print("SUCCESS")
            return Response({'success' : 'OTP SUCCESSFULY Verified!'},status=status.HTTP_200_OK)

        else:
            print("FAILURE")
            return Response({'failure':'OTP Verification Failed!'},status=status.HTTP_400_BAD_REQUEST)

    else:
        return Response({'error':'User Does Not Exist'},status=status.HTTP_400_BAD_REQUEST)

# Create your views here.
