from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Profile
from django.utils import timezone
from .serializers import OTPSerializer
import random

@api_view(['POST',])
def generate_otp(request):

    if request.method == "POST":
        user_profile = None
        phone_number = request.POST.get('phone_number')
        otp = random.randint(1111, 9999)
        serializer = OTPSerializer(data = request.data)

        if not serializer.is_valid():
            return Response({'error':'Phone Number Validation Error!'}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        else:
            try:
                user_profile = Profile.objects.get(phone_number = phone_number)
            except Profile.DoesNotExist:
                user_profile = Profile.objects.create(phone_number = phone_number)
            print(phone_number)
            print(user_profile)
            if user_profile:
                user_profile.otp = otp
                user_profile.save()
                print(otp)
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
            pass

        if user_profile:
            if otp == str(user_profile.otp) and (timezone.now() - user_profile.otp_timestamp).seconds < 30:
                print("SUCCESS")
                data = {'success':'OTP Verified Successfully!', 'OTP':otp}
                return Response(data, status = status.HTTP_200_OK)

            else:
                print("FAILURE")
                return Response({'failure':'OTP Verification Failed!'}, status = status.HTTP_404_NOT_FOUND)

        else:
            return Response({'error':'User Does Not Exist'}, status = status.HTTP_404_NOT_FOUND)
