from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.utils import timezone
import random

from .serializers import OTPSerializer,AddressSerializer,ProfileSerializer
from .models import Profile, Address

@api_view(['POST',])
def generate_otp(request):
    if request.method == "POST":
        user_profile = None
        phone_number = request.POST.get('phone_number')
        otp = random.randint(1111, 9999)
        serializer = OTPSerializer(data = request.data)
        print(request.data)
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

@api_view(['POST',])
def add_address(request):
    if request.method == "POST":
        phone_number = request.POST.get('phone_number')
        otp_serializer = OTPSerializer(data = request.data)

        if not otp_serializer.is_valid():
            return Response({'error':'Validation error!'})

        profile_obj = None

        try:
            profile_obj = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            return Response({'error':'Profile does not exist!'})

        data = request.data.dict()
        data['profile'] = profile_obj.pk
        address_serializer = AddressSerializer(data = data)
        if not address_serializer.is_valid():
            return Response(address_serializer.errors, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        address_serializer.create()
        return Response({'success':'Address saved successfully!','address':address_serializer.data})

@api_view(['PUT',])
def update_address(request):
    if request.method == "PUT":
        address_id = request.POST.get('address_id', None)

        if address_id is None:
            return Response({'error':'Address id not found!'}, status = status.HTTP_404_NOT_FOUND)

        try:
            address_obj = Address.objects.get(pk = address_id)
        except Address.DoesNotExist:
            return Response({'error':'Address does not exist'}, status = status.HTTP_404_NOT_FOUND)

        profile_obj =address_obj.profile
        data = request.data.dict()
        data['profile'] = profile_obj.pk
        address_serializer = AddressSerializer(data = data)
        if not address_serializer.is_valid():
            return Response({'error':address_serializer.errors}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        address_serializer.update(instance = address_obj)
        return Response({'success':'Address updated successfully!','address':address_serializer.data}, status = status.HTTP_200_OK)

@api_view(['PUT',])
def update_profile(request):
    if request.method == "PUT":
        phone_number = request.POST.get('phone_number',None)

        if phone_number is None:
            return Response({'error':'Phone number not provided!'}, status = status.HTTP_400_BAD_REQUEST)

        try :
            profile_obj = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            return Response({'error':'Profile does not exist!'}, status = status.HTTP_404_NOT_FOUND)

        profile_serializer = ProfileSerializer(data = request.data)

        if not profile_serializer.is_valid():
            return Response({'error':profile_serializer.errors}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        profile_serializer.update(instance = profile_obj)
        return Response({'success':'Profile updated successfully!','profile':profile_serializer.data}, status = status.HTTP_200_OK)

@api_view(['POST',])
def retrieve_address(request):
    if request.method == "POST":
        phone_number = request.POST.get('phone_number', None)

        if phone_number is None:
            return Response({'error':'Phone number not provided!'}, status = status.HTTP_400_BAD_REQUEST)

        otp_serializer = OTPSerializer(data = data)

        if not otp_serializer.is_valid():
            return Response({'error':'Validation error!'},status = status.HTTP_404_NOT_FOUND)

        try:
            profile_obj = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            return Response({'error':'Profile does not exist!'}, status = status.HTTP_404_NOT_FOUND)

        retrieved_addresses = Address.objects.filter(profile = profile_obj)
        retrieved_address_serializer = AddressSerializer(retrieved_addresses, many = True)
        return Response({'success':'Phone numbers retrieved successfully!', 'addresses':retrieved_address_serializer.data}, status = status.HTTP_200_OK)
