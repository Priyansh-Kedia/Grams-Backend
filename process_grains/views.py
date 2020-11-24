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
        user_id = request.GET['user_id']
        #scan_serializer = ScanSerializer(Scan.objects.filter(user = Profile.objects.get(pk = user_id)), many = True)
        profile_obj = Profile.objects.get(pk = user_id)
        scan_serializer = ScanSerializer(profile_obj.scan_set.all(), many = True)
        return Response(scan_serializer.data, status = status.HTTP_200_OK)

# Create your views here.
