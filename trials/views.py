
from rest_framework import response
from grams_backend.Constants import FREETRIAL1_DAYS
from grams_backend.celery import basic, prompt_payment_renewal
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from users.models import Profile
from .models import Plan,CurrentStatus
from .serializers import  CurrentStatusSerializer, PlanSerializer,CurrentStatus
from grams_backend.enums import TrialResponse
from datetime import datetime,timedelta
from grams_backend.enums import TrialResponse


@api_view(['GET'])
def plan_status(request,phone_number):
    if request.method == 'GET':
        profile = Profile.objects.get(phone_number = phone_number)
        current,_ = CurrentStatus.objects.get_or_create(user = profile)   
        if not current.name:
            current.name = TrialResponse.TRIAL1
            current.end_date = datetime.now()+timedelta(FREETRIAL1_DAYS)
            current.save()
        current_serializer = CurrentStatusSerializer(current)
        print(current_serializer.data)
        return Response(current_serializer.data, status=status.HTTP_200_OK)

@api_view(['GET'])
def get_all_plans(request):
    plan = Plan.objects.all()
    plan_serializer = PlanSerializer(plan,many = True)
    return Response(plan_serializer.data, status=status.HTTP_200_OK)

@api_view(['POST'])
def update_payment_status(request):
    phone_number = request.POST['phone_number']
    pk = request.POST['id']
    profile = Profile.objects.get(phone_number= phone_number)
    current = CurrentStatus.objects.get(user = profile)
    paid_plan = Plan.objects.get(pk =pk)
    current.plan = paid_plan
    current.end_date = datetime.now()+timedelta(paid_plan.no_of_days)
    current.no_of_readings = current.no_of_readings+ paid_plan.readings
    current.name = TrialResponse.PAID
    current.save()
    prompt_payment_renewal.apply_async(args = [phone_number],countdown =  paid_plan.no_of_days*86400)
    current_serializer = CurrentStatusSerializer(current)
    return Response(current_serializer.data,status=status.HTTP_200_OK)
