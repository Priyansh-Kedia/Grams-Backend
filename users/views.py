from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.utils import timezone
from django.forms.models import model_to_dict
from . import utils
from .serializers import OTPSerializer,AddressSerializer,ProfileSerializer, ImageSerializer
from process_grains.serializers import ScanSerializer
from main import main
from django_q.tasks import async_task
from .models import Profile, Address, Image
from process_grains.models import Scan
from grams_backend import Constants
from .utils import run_ml_code
from rest_framework.decorators import parser_classes
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from mock import mock
from datetime import datetime 

import requests
from urllib.parse import unquote


# OTP #
@api_view(['POST',])
def generate_otp(request):
    if request.method == "POST":
        hashValue = request.POST.get(Constants.HASH)
        phone_number = request.POST.get(Constants.PHONE_NUMBER)         
        otp = utils.otp_generator()
        serializer = OTPSerializer(data = request.data)

        if not serializer.is_valid():
            return Response({Constants.MESSAGE:"Invalid phone number", Constants.PROFILE:None, Constants.IS_VERIFIED:False}, status = status.HTTP_200_OK)

        try:
            user_profile = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            user_profile = Profile.objects.create(phone_number = phone_number)

        if phone_number == '+911111111111':
            user_profile.otp = '1234'
            user_profile.save()
            message = Constants.GRAMS_MESSAGE+" {otp} \n {hash}".format(otp = otp, hash = hashValue)
            data = {Constants.MESSAGE:message, Constants.PROFILE:model_to_dict(user_profile), Constants.IS_VERIFIED:False}
            return Response(data, status = status.HTTP_200_OK) 
        url = Constants.OTP_URL+ Constants.OTP_KEY+ "SMS/" + phone_number + "/" + str(otp)
        requests.post( url )

        if user_profile:
            user_profile.otp = otp
            user_profile.save()          
            message = Constants.GRAMS_MESSAGE+" {otp} \n {hash}".format(otp = otp, hash = hashValue)
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


# Profile #
@api_view(['PUT',])
def update_profile(request):
    if request.method == "PUT":
        profile_id = request.data.get('profile_id')
        profile_serializer = ProfileSerializer(data =request.data)

        if not profile_serializer.is_valid():
            return Response({Constants.MESSAGE:profile_serializer.errors}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        updated_profile = profile_serializer.update(instance = Profile.objects.get(profile_id = profile_id))
        return Response(model_to_dict(updated_profile), status = status.HTTP_200_OK)

@api_view(['GET',])
def retrieve_profile(request):
    if request.method == "GET":
        phone_number = request.GET.get(Constants.PHONE_NUMBER, None)

        try:
            profile_obj = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            return Response({Constants.MESSAGE:'Profile does not exist!'}, status = status.HTTP_404_NOT_FOUND)

        return Response(model_to_dict(profile_obj), status = status.HTTP_200_OK)



# Address #
@api_view(['POST',])
def add_address(request):
    if request.method == "POST":
        add_obj = Address(profile_id=Profile.objects.get(pk=1), address= 'TESTING')
        print(add_obj.address)
        add_obj.save()
        address_serializer = AddressSerializer(data = request.data)
        
        if not address_serializer.is_valid():
            return Response(address_serializer.errors, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        updated_address_obj = address_serializer.create()
        return Response(model_to_dict(updated_address_obj), status = status.HTTP_200_OK)

@api_view(['PUT',])
def update_address(request):
    if request.method == "PUT":
        address_serializer = AddressSerializer(data = request.data)

        if not address_serializer.is_valid():
            return Response({Constants.MESSAGE:address_serializer.errors}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)

        updated_address = address_serializer.update(instance = Address.objects.get(pk = request.data['address_id']))
        return Response(model_to_dict(updated_address), status = status.HTTP_200_OK)

@api_view(['GET',])
def retrieve_address(request):
    if request.method == "GET":
        profile_id = request.GET.get(Constants.PROFILE_ID, None)
        retrieved_addresses = Profile.getAllAddresses(profile_id)

        retrieved_address_serializer = AddressSerializer(retrieved_addresses, many = True)
        return Response(retrieved_address_serializer.data, status = status.HTTP_200_OK)




@api_view(['GET',])
def health(request):
    return Response("OK", status = status.HTTP_200_OK)


@api_view(['POST',])
@parser_classes([MultiPartParser, FormParser])
def upload_image(request, phone_number):
    if request.method == 'POST':

        
        profile = Profile.objects.get(phone_number = phone_number)
        
        
 
        image_obj = Image.objects.create(image = request.POST['image'])
        print(image_obj.image.url)
        async_task(run_ml_code)
    



        heading_msg = "Your results will be available soon"
        content_msg = "Your results will come soon"
        data = {"app_id": Constants.APP_ID, "contents": {"en": content_msg}, "headings": {"en": heading_msg}, "include_external_user_ids": [phone_number] , "chrome_web_image": "https://images.ctfassets.net/hrltx12pl8hq/7yQR5uJhwEkRfjwMFJ7bUK/dc52a0913e8ff8b5c276177890eb0129/offset_comp_772626-opt.jpg?fit=fill&w=800&h=300"}

        requests.post(Constants.API_URL,headers={"Authorization": "Basic "+Constants.API_KEY}, json=data)

        data = {
            'data': 'hello',
        }
        return Response(data, status = status.HTTP_200_OK)

class MyImageView(APIView):
		def post(self, request, *args, **kwargs):
				file_serializer = ImageSerializer(data=request.data)
				if file_serializer.is_valid():
						file_serializer.save()
						return Response(file_serializer.data, status=status.HTTP_201_CREATED)
				else:
						return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)