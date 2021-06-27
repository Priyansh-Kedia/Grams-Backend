from rest_framework import serializers
from .models import Plan,CurrentStatus

class PlanSerializer(serializers.ModelSerializer):

    class Meta:
        model = Plan
        fields = '__all__'

class CurrentStatusSerializer(serializers.ModelSerializer):

    class Meta:
        model = CurrentStatus
        fields = '__all__'
