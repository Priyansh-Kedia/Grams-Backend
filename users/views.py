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
from grams_backend import Constants


# OTP #
# ======================================================================================================================================= #
@api_view(['POST',])
def generate_otp(request):
    if request.method == "POST":
        hashValue = request.POST.get(Constants.HASH)
        phone_number = request.POST.get(Constants.PHONE_NUMBER)         
        otp = random.randint(1111, 9999)
        serializer = OTPSerializer(data = request.data)

        if not serializer.is_valid():
            return Response({Constants.MESSAGE:"Invalid phone number", Constants.PROFILE:None, Constants.IS_VERIFIED:False}, status = status.HTTP_200_OK)

        try:
            user_profile = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            user_profile = Profile.objects.create(phone_number = phone_number)

        if user_profile:
            user_profile.otp = otp
            user_profile.save()          
            message = "<#> Your GramsApp code is: {otp} \n {hash}".format(otp = otp, hash = hashValue)
            data = {Constants.MESSAGE:message, Constants.PROFILE:model_to_dict(user_profile), Constants.IS_VERIFIED:False}
            return Response(data, status = status.HTTP_200_OK)
        else:
            return Response({Constants.MESSAGE:'User Does Not Exist', Constants.IS_VERIFIED:False, Constants.PROFILE:None}, status = status.HTTP_200_OK)

@api_view(['POST',])
def verify_otp(request):
    if request.method == "POST":
        otp = request.POST.get(Constants.OTP)
        phone_number = request.POST.get(Constants.PHONE_NUMBER)
        serializer = OTPSerializer(data=request.data)

        if not serializer.is_valid():
            return Response({Constants.MESSAGE:'Phone Number Not Validated!', Constants.PROFILE:None, Constants.IS_VERIFIED:False}, status = status.HTTP_200_OK)

        try:
            user_profile = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            return Response({Constants.MESSAGE:'Profile does not exist!', Constants.PROFILE:None, Constants.IS_VERIFIED:False}, status = status.HTTP_200_OK)

        if user_profile:
            if otp == str(user_profile.otp) :
                if (timezone.now() - user_profile.otp_timestamp).seconds < 1800:
                    data = {Constants.MESSAGE:'OTP Verified Successfully!', Constants.PROFILE:model_to_dict(user_profile), Constants.IS_VERIFIED:True}
                    return Response(data, status = status.HTTP_200_OK)
                else:
                    return Response({Constants.MESSAGE:'OTP has expired!', Constants.PROFILE:model_to_dict(user_profile), Constants.IS_VERIFIED:False}, status = status.HTTP_200_OK)
            else:
                return Response({Constants.MESSAGE:'OTP Verification Failed!. The entered OTP is incorrect', Constants.PROFILE:model_to_dict(user_profile), Constants.IS_VERIFIED:False}, status = status.HTTP_200_OK)

# ======================================================================================================================================= #

# Profile #
# ======================================================================================================================================= #

@api_view(['PUT',])
def update_profile(request):
    if request.method == "PUT":
        profile_serializer = ProfileSerializer(data = request.data)
        print(request.data['profile_id'])
        if not profile_serializer.is_valid():
            return Response({Constants.MESSAGE:profile_serializer.errors}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)
        print(profile_serializer.data)
        print(repr(profile_serializer))
        #if request.data['designation'] is None:
        #    print("HA")
        updated_profile = profile_serializer.update(instance = Profile.objects.get(pk = request.data['profile_id']))
        return Response(model_to_dict(updated_profile), status = status.HTTP_200_OK)

@api_view(['GET',])
def retrieve_profile(request):
    if request.method == "GET":
        profile_id = request.GET.get(Constants.PHONE_NUMBER, None)
        profile_obj = Profile.objects.get(phone_number = profile_id)
        profile_serializer = ProfileSerializer(data = model_to_dict(profile_obj))
        
        # if not profile_serializer.is_valid():       
        #     print(profile_serializer.errors)    
        #     return Response(profile_serializer.errors, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        # updated_dict={Constants.ID:profile_id}
        # updated_dict.update(profile_serializer.data)
        # print(profile_serializer.data)
        return Response(model_to_dict(profile_obj), status = status.HTTP_200_OK)

# ======================================================================================================================================= #


# Address #
# ======================================================================================================================================= #

@api_view(['POST',])
def add_address(request):
    if request.method == "POST":   
        address_serializer = AddressSerializer(data = request.data)

        if not address_serializer.is_valid():
            return Response(address_serializer.errors, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        updated_address_obj = address_serializer.create()
        return Response({Constants.MESSAGE:'Address saved successfully!', Constants.ADDRESS:model_to_dict(updated_address_obj)}, status = status.HTTP_200_OK)

@api_view(['PUT',])
def update_address(request):
    if request.method == "PUT":
        #address_id = request.POST.get(Constants.ADDRESS_ID, None)
        #address_obj = Address.objects.get(pk = address_id)
        address_serializer = AddressSerializer(data = request.data)
        
        if not address_serializer.is_valid():
            print(address_serializer.errors)
            return Response({Constants.MESSAGE:address_serializer.errors}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)
        
        updated_address = address_serializer.update(instance = Address.objects.get(pk = request.data['id']))
        return Response(model_to_dict(updated_address), status = status.HTTP_200_OK)

@api_view(['GET',])
def retrieve_address(request):
    if request.method == "GET":
        profile_id = request.GET.get(Constants.PROFILE_ID, None)

        try:
            profile_obj = Profile.objects.get(pk = profile_id)
        except Profile.DoesNotExist:
            return Response({Constants.MESSAGE:'Profile does not exist!'}, status = status.HTTP_404_NOT_FOUND)
       
        retrieved_addresses = Address.objects.filter(profile_id = profile_obj.pk)
        retrieved_address_serializer = AddressSerializer(retrieved_addresses, many = True)
        print(retrieved_address_serializer.data)
        return Response(retrieved_address_serializer.data, status = status.HTTP_200_OK)

# ======================================================================================================================================= #