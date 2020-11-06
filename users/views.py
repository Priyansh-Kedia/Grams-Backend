from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.utils import timezone
import random
from django.forms.models import model_to_dict


from .serializers import OTPSerializer,AddressSerializer,ProfileSerializer
from .models import Profile, Address
from . import Constants


@api_view(['POST',])
def generate_otp(request):
    if request.method == "POST":
        user_profile = None
        phone_number = request.POST.get('phone_number')
        otp = random.randint(1111, 9999)
        serializer = OTPSerializer(data = request.data)
        print(request.data)
        if not serializer.is_valid():
            return Response({Constants.ERROR:'Phone Number Validation Error!', Constants.PHONE_NUMBER:phone_number}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        try:
            user_profile = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            user_profile = Profile.objects.create(phone_number = phone_number)

        if user_profile:
            user_profile.otp = otp
            user_profile.save()
            data = {Constants.MESSAGE:'OTP Generated Successfully!', 'OTP':otp, Constants.PROFILE:model_to_dict(user_profile)}#add profile also
            return Response(data, status = status.HTTP_200_OK)
        else:
            return Response({Constants.ERROR:'User Does Not Exist'}, status = status.HTTP_404_NOT_FOUND)
#create constant message error 
#Dont send OTP
#is verified boolean True False
#profilr error None ad for success Profile
@api_view(['POST',])
def verify_otp(request):
    if request.method == "POST":
        otp = request.POST.get('otp')
        phone_number = request.POST.get('phone_number')
        serializer = OTPSerializer(data=request.data)

        if not serializer.is_valid():
            return Response({Constants.MESSAGE:'Phone Number Not Validated!', Constants.PROFILE:'None', Constants.PHONE_NUMBER:phone_number, Constants.IS_VERFIED:False}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        user_profile = None

        try:
            user_profile = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            return Response({Constants.MESSAGE:'None', Constants.PROFILE:'None', Constants.IS_VERFIED:False})

        if user_profile:
            if otp == str(user_profile.otp) :
                if (timezone.now() - user_profile.otp_timestamp).seconds < 10:
                    data = {Constants.MESSAGE:'OTP Verified Successfully!', Constants.PROFILE:model_to_dict(user_profile), Constants.IS_VERIFIED:True}
                    return Response(data, status = status.HTTP_200_OK)
                else:
                    return Response({Constants.MESSAGE:'OTP has expired!'}, status = status.HTTP_404_NOT_FOUND)
            else:
                return Response({Constants.MESSAGE:'OTP Verification Failed!. The entered OTP is incorrect', Constants.PROFILE:'None', Constants.IS_VERIFIED:False}, status = status.HTTP_404_NOT_FOUND)

@api_view(['POST',])
def add_address(request):
    if request.method == "POST":
        phone_number = request.POST.get('phone_number')
        otp_serializer = OTPSerializer(data = request.data)

        if not otp_serializer.is_valid():
            return Response({Constants.ERROR:'Validation error!', Constants.PHONE_NUMBER:phone_number})

        profile_obj = None

        try:
            profile_obj = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            return Response({Constants.ERROR:'Profile does not exist!', Constants.PHONE_NUMBER:phone_number})

        data = request.data.dict()
        data['profile'] = profile_obj.pk
        address_serializer = AddressSerializer(data = data)
        if not address_serializer.is_valid():
            return Response(address_serializer.errors, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        address_serializer.create()
        return Response({Constants.SUCCESS:'Address saved successfully!','address':address_serializer.data})

@api_view(['PUT',])
def update_address(request):
    if request.method == "PUT":
        address_id = request.POST.get('address_id', None)

        if address_id is None:
            return Response({Constants.ERROR:'Address id not found!'}, status = status.HTTP_404_NOT_FOUND)

        try:
            address_obj = Address.objects.get(pk = address_id)
        except Address.DoesNotExist:
            return Response({Constants.ERROR:'Address does not exist'}, status = status.HTTP_404_NOT_FOUND)

        profile_obj =address_obj.profile
        data = request.data.dict()
        data['profile'] = profile_obj.pk
        address_serializer = AddressSerializer(data = data)
        if not address_serializer.is_valid():
            return Response({Constants.ERROR:address_serializer.errors}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        updated_address = address_serializer.update(instance = address_obj)
        return Response({Constants.SUCCESS:'Address updated successfully!','address':model_to_dict(updated_address)}, status = status.HTTP_200_OK)

@api_view(['PUT',])
def update_profile(request):
    if request.method == "PUT":
        phone_number = request.POST.get('phone_number',None)

        if phone_number is None:
            return Response({Constants.ERROR:'Phone number not provided!'}, status = status.HTTP_400_BAD_REQUEST)

        try :
            profile_obj = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            return Response({Constants.ERROR:'Profile does not exist!', Constants.PHONE_NUMBER:phone_number}, status = status.HTTP_404_NOT_FOUND)

        profile_serializer = ProfileSerializer(data = request.data)

        if not profile_serializer.is_valid():
            return Response({Constants.ERROR:profile_serializer.errors}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        updated_profile = profile_serializer.update(instance = profile_obj)
        return Response({Constants.SUCCESS:'Profile updated successfully!','profile':model_to_dict(updated_profile)}, status = status.HTTP_200_OK)

@api_view(['POST',])
def retrieve_address(request):
    if request.method == "POST":
        phone_number = request.POST.get('phone_number', None)

        if phone_number is None:
            return Response({Constants.ERROR:'Phone number not provided!'}, status = status.HTTP_400_BAD_REQUEST)

        otp_serializer = OTPSerializer(data = request.data)

        if not otp_serializer.is_valid():
            return Response({Constants.ERROR:'Validation error!', Constants.PHONE_NUMBER:phone_number},status = status.HTTP_404_NOT_FOUND)

        try:
            profile_obj = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            return Response({Constants.ERROR:'Profile does not exist!'}, status = status.HTTP_404_NOT_FOUND)

        retrieved_addresses = Address.objects.filter(profile = profile_obj)
        retrieved_address_serializer = AddressSerializer(retrieved_addresses, many = True)
        return Response({Constants.SUCCESS:'Phone numbers retrieved successfully!', 'addresses':retrieved_address_serializer.data}, status = status.HTTP_200_OK)
