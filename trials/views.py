from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from users.models import Profile
from .models import FreeTrial,Paid,Plan
from .serializers import PaidSerializer, PlanSerializer
from grams_backend.celery import prompt_payment_renewal
from datetime import datetime,timedelta
from grams_backend.enums import TrialResponse
from grams_backend.celery import basic

@api_view(['GET'])
def plan_status(request,phone_number):
    if request.method == 'GET':
        profile = Profile.objects.get(phone_number = phone_number)
        data  = {}
        try:
            free_trial = FreeTrial.objects.get(user = profile)
            print(free_trial.second_trial)
            if free_trial.first_trial and free_trial.second_trial:
                try:
                    paid = profile.user
                    data ['plan'] = TrialResponse.PAID
                    data['start_date'] = paid.start_date
                    data['end_date'] = paid.end_date 
                except:
                    data['plan'] = TrialResponse.TRIAL2
                    data['start_date'] = free_trial.start_date
                    data['end_date'] = free_trial.end_date 
            elif free_trial.first_trial:
                data['plan'] = TrialResponse.TRIAL1
                data['start_date'] = free_trial.start_date
                data['end_date'] = free_trial.end_date     
        except:
            free_trial = FreeTrial.objects.create(user = profile)
            basic.apply_async(args = [phone_number],countdown =  free_trial.t1*86400)  
            data ['plan'] = TrialResponse.TRIAL1 
        return Response(data, status=status.HTTP_200_OK)

@api_view(['GET'])
def get_all_plans(request):
    plan = Plan.objects.all()
    plan_serializer = PlanSerializer(plan,many = True)
    return Response(plan_serializer.data, status=status.HTTP_200_OK)

@api_view(['POST'])
def update_payment_status(request):
    phone_number = request.POST['phone_number']
    profile = Profile.objects.get(phone_number= phone_number)
    paid, _ = Paid.objects.get_or_create(user =profile)
    paid.paid = True
    paid.start_date = datetime.now()
    paid.end_date = paid.start_date + timedelta(days=30)
    paid.save()
    prompt_payment_renewal.apply_async(args = [phone_number],countdown =  2592000)
    paid_serializer = PaidSerializer(paid)
    return Response(paid_serializer.data, status=status.HTTP_200_OK)