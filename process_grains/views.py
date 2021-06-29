from grams_backend import Constants
from typing import Optional
from django.shortcuts import render
from rest_framework import status
from rest_framework import response
from rest_framework.response import Response
from rest_framework.decorators import api_view

from users.models import Profile
from .models import Scan, SubType, Type 
from .serializers import ScanSerializer, SubTypeSerializer, TypeSerializer


@api_view(['GET',])
def retreive_scan(request):
    if request.method == "GET":
        user_id = request.GET['profile_id']
        scan_set = Profile.getAllScans(user_id)
        scan_serializer = ScanSerializer(scan_set, many = True)
        return Response(scan_serializer.data, status = status.HTTP_200_OK)

@api_view(['POST',])
def delete_readings(request):
    scan_id = request.POST.get('scan_id')
    try:
        Scan.objects.get(scan_id=scan_id).delete()
        return Response(status = status.HTTP_200_OK)
    except:
        return Response(status = status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def get_all_types(request):
    all_types = Type.objects.all()
    all_types = TypeSerializer(all_types, many = True)
   
    return Response(all_types.data, status = status.HTTP_200_OK)


@api_view(['GET'])
def get_subtype(request):
    id = request.GET['id']
    if id:
        try:
            type_obj = Type.objects.get(id = id)
        except:
            return Response({Constants.MESSAGE:'Type does not exist'},status = status.HTTP_404_NOT_FOUND)
        
        all_subtypes = SubType.objects.filter(type = type_obj)
    
    else:
        all_subtypes = SubType.objects.all()
    
    all_subtypes = SubTypeSerializer(all_subtypes,many=True)
    
    return Response(all_subtypes.data,status= status.HTTP_200_OK)