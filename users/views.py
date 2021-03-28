from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.utils import timezone
import random
from django.forms.models import model_to_dict

from .serializers import OTPSerializer,AddressSerializer,ProfileSerializer, ImageSerializer
from .models import Profile, Address, Image
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

# ======================================================================================================================================= #


# Address #
# ======================================================================================================================================= #

@api_view(['POST',])
def add_address(request):
    if request.method == "POST":
        add_obj = Address(profile_id=Profile.objects.get(pk=1), address= 'TESTING')
        print(add_obj.address)
        add_obj.save()
        #print(add_obj.address)  
        #add_obj_1 = Address.objects.create(profile_id=Profile.objects.get(pk=1), address= 'kvgdsn')
        #print(add_obj.address)
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


# ======================================================================================================================================= #
from rest_framework.decorators import parser_classes
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
# from py_source import py_main

from grams_backend.py_source import py_main
import requests
@api_view(['POST',])
@parser_classes([MultiPartParser, FormParser])
def upload_image(request):
    if request.method == 'POST':
        #print(request.data)
        #image_obj = Image.objects.create(image = request.data['image'])
        image_serializer = ImageSerializer(data = request.data)
        data = {    "app_id": "fad6e42a-0b02-45d6-9ab0-a654b204aca9",    "included_segments": ["All"],    "contents": {"en": "Hello"}}

        requests.post(    "https://onesignal.com/api/v1/notifications",    headers={"Authorization": "Basic NDJkOGMyZDQtMjgyYi00Y2JkLWFjZTgtZGQ2NjQ1NDUwNzg3"}, json=data)
        # print(py_source.CSV_name)
        #print(model_to_dict(image_obj))
        # run py_source.py
        # py_main()
        if not image_serializer.is_valid():
                return Response(image_serializer.errors)
        else:
            image_serializer.save() 
            print(image_serializer.data)
        #  image_serializer.data['image']
        # csv_file = py_main(Image = request.data['image'] , Rescale_Factor = 1, Diameter = 20)# 20 for wheat rf = 1 if image is small
        #print(repr(image_serializer))
        # print(image_serializer.validated_data['image'])
        #print(image_obj.image)
        data = {
            'data': image_serializer.data,
            # 'csv': csv_file
        }
        return Response(data, status = status.HTTP_200_OK)

class MyImageView(APIView):
		# MultiPartParser AND FormParser
		# https://www.django-rest-framework.org/api-guide/parsers/#multipartparser
		# "You will typically want to use both FormParser and MultiPartParser
		# together in order to fully support HTML form data."
		#parser_classes = (MultiPartParser, FormParser)
		def post(self, request, *args, **kwargs):
				file_serializer = ImageSerializer(data=request.data)
				if file_serializer.is_valid():
						file_serializer.save()
						return Response(file_serializer.data, status=status.HTTP_201_CREATED)
				else:
						return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)