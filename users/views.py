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

@api_view(['PUT'],)
def update(request):
    if request.method == "PUT":
        address_id = request.POST.get('address_id')
        profile_serializer = ProfileSerializer(data = request.data)
        if not profile_serializer.is_valid():
            return Response(profile_serializer.errors,status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        try:
            address_obj = Address.objects.get(pk = address_id)
        except Address.DoesNotExist:
            return Response({'error':'Address not found!'}, status = status.HTTP_404_NOT_FOUND)

        profile_obj = address_obj.profile
        data = request.data.dict()
        data['profile'] = profile_obj.pk
        address_serializer = AddressSerializer(data = data)

        if not address_serializer.is_valid():
            return Response({'error':address_serializer.errors}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        address = request.POST.get('address', address_obj.address)
        city = request.POST.get('city', address_obj.city)
        state = request.POST.get('state', address_obj.state)
        country = request.POST.get('country', address_obj.country)
        name = request.POST.get('name', profile_obj.name)
        phone_number = request.POST.get('phone_number', profile_obj.phone_number)
        company_name = request.POST.get('company_name', profile_obj.company_name)
        designation = request.POST.get('designation', profile_obj.designation)
        email_id = request.POST.get('email_id', profile_obj.email_id)

        address_obj.city = city
        address_obj.country = country
        address_obj.state = state
        address_obj.address = address
        profile_obj.name = name
        profile_obj.phone_number = phone_number
        profile_obj.company_name = company_name
        profile_obj.designation = designation
        profile_obj.email_id = email_id

        profile_obj.save()
        address_obj.save()
        #data = {'sucess':'Fields Updated Successfully!', 'company_name':company_name, 'city':city, 'name':name, 'phone_number':phone_number, 'address':address, 'email_id':email_id}
        data = {'success':'Fields updated successfully!','address':address_serializer.data}
        return Response(data)

@api_view(['POST',])
def retrieve_address(request):
    if request.method == "POST":
        phone_number = request.POST.get('phone_number', None)

        if phone_number is None:
            return Response({'error':'Phone number not provided!'}, status = status.HTTP_400_BAD_REQUEST)

        otp_serializer = OTPSerializer(data = request.data)

        if not otp_serializer.is_valid():
            return Response({'error':'Validation error!'},status = status.HTTP_404_NOT_FOUND)

        try:
            profile_obj = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            return Response({'error':'Profile does not exist!'}, status = status.HTTP_404_NOT_FOUND)

        retrieved_addresses = Address.objects.filter(profile = profile_obj)
        retrieved_address_serializer = AddressSerializer(retrieved_addresses, many = True)
        return Response({'success':'Phone numbers retrieved successfully!', 'addresses':retrieved_address_serializer.data}, status = status.HTTP_200_OK)
