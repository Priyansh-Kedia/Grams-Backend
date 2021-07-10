from grams_backend.enums import TrialResponse
from re import T, sub
import re
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
from .models import Feedback, Profile, Address, Image
from process_grains.models import Scan
from grams_backend import Constants
from rest_framework.decorators import parser_classes
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from mock import mock
from datetime import date, datetime, timedelta
import requests
from urllib.parse import unquote
from .tasks import add, run_ml_code
from trials.serializers import PlanSerializer,CurrentStatusSerializer
from trials.models import CurrentStatus
from decouple import config

import users

from users import models


# OTP #
@api_view(['POST',])
def generate_otp(request):
    if request.method == "POST":
        hashValue = request.POST.get(Constants.HASH)
        phone_number = request.POST.get(Constants.PHONE_NUMBER)    
        print(phone_number)     
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
        url = Constants.OTP_URL+ config(Constants.OTP_KEY)+ "SMS/" + phone_number + "/" + str(otp)
        print(url)
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
                    if user_profile.is_new_user:
                        current = CurrentStatus.objects.create(user = user_profile)   
                        if not current.name:
                            current.name = TrialResponse.TRIAL1
                            current.end_date = datetime.now()+timedelta(Constants.FREETRIAL1_DAYS)
                            current.save()
                    user_profile.is_new_user = False
                    user_profile.save()
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
        print(updated_profile.name)
        current = CurrentStatus.objects.get(user = updated_profile)
        if updated_profile.gst_no and current.name == TrialResponse.TRIAL1:
            current.name = TrialResponse.TRIAL2
            current.end_date = current.end_date + timedelta(Constants.FREETRIAL2_DAYS)
            current.no_of_readings += Constants.FREETRIAL2_READINGS
            current.save()
        print(current.name)
        profile_dict = model_to_dict(updated_profile)
        profile_dict["end_date"] = current.end_date
        profile_dict["current_status_name"] = current.name
        return Response(profile_dict, status = status.HTTP_200_OK)

@api_view(['GET',])
def retrieve_profile(request):
    if request.method == "GET":
        phone_number = request.GET['phone_number']
        print(phone_number)
        try:
            profile_obj = Profile.objects.get(phone_number = phone_number)
        except Profile.DoesNotExist:
            return Response({Constants.MESSAGE:'Profile does not exist!'}, status = status.HTTP_404_NOT_FOUND)
        return Response(model_to_dict(profile_obj), status = status.HTTP_200_OK)



# Address #
@api_view(['POST',])
def add_address(request):
    if request.method == "POST":
        address_serializer = AddressSerializer(data = request.data)
        if not address_serializer.is_valid():
            return Response(address_serializer.errors, status = status.HTTP_422_UNPROCESSABLE_ENTITY)
        address = address_serializer.create()
        print(model_to_dict(address))
        return Response(model_to_dict(address), status = status.HTTP_200_OK)

@api_view(['PUT',])
def update_address(request):
    if request.method == "PUT":
        print(request.data)
        address_serializer = AddressSerializer(data = request.data)
        if not address_serializer.is_valid():
            return Response({Constants.MESSAGE:address_serializer.errors}, status = status.HTTP_422_UNPROCESSABLE_ENTITY)
        updated_address = address_serializer.update(instance = Address.objects.get(pk = request.data['address_id']))
        return Response(model_to_dict(updated_address), status = status.HTTP_200_OK)

@api_view(['GET',])
def retrieve_address(request):
    if request.method == "GET":
        profile_id = request.GET.get(Constants.PROFILE_ID, None)
        profile = Profile.objects.get(pk = profile_id)
        try:
            retrieved_addresses = Address.objects.get(profile_id = profile)
            retrieved_address_serializer = AddressSerializer(retrieved_addresses)
            print(retrieved_address_serializer.data)
            return Response(retrieved_address_serializer.data, status = status.HTTP_200_OK)
        except Address.DoesNotExist:
            return Response({Constants.MESSAGE: "Does not exist"}, status= status.HTTP_400_BAD_REQUEST)


        

@api_view(['GET',])
def health(request):
    return Response("OK", status = status.HTTP_200_OK)


@api_view(['POST',])
@parser_classes([MultiPartParser, FormParser])
def upload_image(request, phone_number, type, sub_type):
    if request.method == 'POST':
        # phone_number = request.POST['phone_number']
        profile = Profile.objects.get(phone_number = phone_number)
        image_obj = Image.objects.create(image = request.data['image'])
        # item_type = request.POST['type']
        # sub_type = request.POST['sub_type']
        item_type = type
        print(image_obj.image.url, phone_number)
        run_ml_code.delay(phone_number,image_obj.image.url,item_type,sub_type)
        # heading_msg = "Your results will be available soon"
        # content_msg = "Your results will come soon"
        # data = {"app_id": Constants.APP_ID, "contents": {"en": content_msg}, "headings": {"en": heading_msg}, "include_external_user_ids": [phone_number] , "chrome_web_image": Constants.CHROME_WEB_IMAGE}
        # requests.post(Constants.API_URL,headers={"Authorization": "Basic "+Constants.API_KEY}, json=data)
        current = CurrentStatus.objects.get(user = profile)
        current.no_of_readings -= 1
        current.save()
        current_serializer = CurrentStatusSerializer(current)
        return Response(current_serializer.data, status = status.HTTP_200_OK)

@api_view(['POST'])
def feedback(request):
    if request.method == "POST":
        feedback = request.POST.get(Constants.FEEDBACK)
        profile_id = request.POST.get(Constants.PROFILE_ID)
        try:
            profile_obj = Profile.objects.get(profile_id = profile_id)
        except:
            return Response({Constants.MESSAGE:'Profile does not exist'}, status = status.HTTP_404_NOT_FOUND)
        feedback_obj = Feedback.objects.create(feedback = feedback, user = profile_obj)
        feedback_obj = model_to_dict(feedback_obj)
        data = {
            Constants.MESSAGE: "Feedback received successfully",
            Constants.FEEDBACK : feedback_obj, 
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