from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view

from users.models import Profile
from .models import Scan
from .serializers import ScanSerializer


@api_view(['GET',])
def retreive_scan(request):
    if request.method == "GET":
        user_id = request.GET['profile_id']
        # print(request.GET['phone_number'])
        # phone_number = '+' + request.GET['phone_number']
        # print(phone_number)
        scan_set = Profile.getAllScans(user_id)
        # print(scan_set)
        scan_serializer = ScanSerializer(scan_set, many = True)
        return Response(scan_serializer.data, status = status.HTTP_200_OK)

# Create your views here.
