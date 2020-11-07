from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.utils import timezone
import datetime

from .models import Subscription, Plan
from users.models import Profile
from .serializers import PlanSerializer
from grams_backend import Constants


@api_view(['GET',])
def my_plans(request):
    if request.method == "GET":
        plans = Plan.objects.all()
        plan_serializer = PlanSerializer(plans, many = True)
        #if not plan_serializer.is_valid():
        #    return Response(plan_serializer.errors, status = status.HTTP_200_OK)
        data = {Constants.PLANS:plan_serializer.data}
        return Response(data, status = status.HTTP_200_OK)

# Create your views here.
@api_view(['POST',])
def add_subscription(request):
    if request.method == "POST":
        profile_id = request.POST['profile_id']
        no_of_days = request.POST['no_of_days']

        try:
            profile_obj = Profile.objects.get(pk = profile_id)
        except Profile.DoesNotExist:
            return Response({Constants.MESSAGE:'Profile does not exist!'}, status = status.HTTP_200_OK)
        
        subscription_obj = Subscription.objects.get(profile = profile_obj)
        subscription_obj.expiry_date += datetime.timedelta(days=int(no_of_days))
        subscription_obj.save()
        return Response({Constants.MESSAGE:'Subscription added successfully!', Constants.EXPIRY_DATE:subscription_obj.expiry_date.date()}, status = status.HTTP_200_OK)