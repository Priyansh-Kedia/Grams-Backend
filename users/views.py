from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.utils import timezone
import random

from .serializers import OTPSerializer
from .models import Profile

@api_view(['POST',])
def generate_otp(request):
    if request.method == "POST":
        user_profile = None
        phone_number = request.POST.get('phone_number')
        otp = random.randint(1111, 9999)
        serializer = OTPSerializer(data = request.data)

        if not serializer.is_valid():
            return Response({'error':'Phone Number Validation Error!'}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        try:
            user_profile = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            user_profile = Profile.objects.create(phone_number = phone_number)

        if user_profile:
            user_profile.otp = otp
            user_profile.save()
            data = {'success':'OTP Generated Successfully!', 'OTP':otp}
            return Response(data, status = status.HTTP_200_OK)
        else:
            return Response({'error':'User Does Not Exist'}, status = status.HTTP_404_NOT_FOUND)

@api_view(['POST',])
def verify_otp(request):
    if request.method == "POST":
        otp = request.POST.get('otp')
        phone_number = request.POST.get('phone_number')
        serializer = OTPSerializer(data=request.data)

        if not serializer.is_valid():
            return Response({'error':'Phone Number Not Validated!'}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        user_profile = None

        try:
            user_profile = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            return Response({'error':'Please Generate OTP First!'})

        if user_profile:
            if otp == str(user_profile.otp) :
                if (timezone.now() - user_profile.otp_timestamp).seconds < 10:
                    data = {'success':'OTP Verified Successfully!', 'OTP':otp}
                    return Response(data, status = status.HTTP_200_OK)
                else:
                    return Response({'failure':'OTP has expired!'}, status = status.HTTP_404_NOT_FOUND)
            else:
                return Response({'failure':'OTP Verification Failed!. The entered OTP is incorrect'}, status = status.HTTP_404_NOT_FOUND)
