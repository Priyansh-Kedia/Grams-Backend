from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import HttpResponse,JsonResponse
from django.utils import timezone
import datetime
import csv
from django.core.files import File
import os
#2os.chdir("path/to")


from grams_backend import Constants
from .models import Profile
from .serializers import GrainSerializer


@api_view(['POST',])
def add_grain(request):
    if request.method == "POST":
        profile_id = request.POST.get('profile_id')
        image = request.data.get('image')
        image_name = str(image)

        try:
            profile_obj = Profile.objects.get(pk = profile_id)
        except Profile.DoesNotExist:
            return Response({Constants.MESSAGE:'Profile does not exist!'}, status = status.HTTP_200_OK)

        name = profile_obj.name  
        filename = "{name}_{datetime}".format(name = name, datetime = timezone.now())

        data = request.data.dict()
        data['profile'] = profile_id           
        data['csv_file'] = File(open("grains/csv_file.csv")) 
        grain_serializer = GrainSerializer(data = data)
        
        if not grain_serializer.is_valid():
            return Response(grain_serializer.errors, status = status.HTTP_200_OK)
        
        #return Response(grain_serializer.data, status = status.HTTP_200_OK)
        return Response({'file':File(open("grains/csv_file.csv"))}, content_type="text/csv", status = status.HTTP_200_OK)
        
       
        