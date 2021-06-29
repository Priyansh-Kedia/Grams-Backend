from rest_framework import response, serializers
from .models import Plan,CurrentStatus
from users.serializers import ProfileSerializer

class PlanSerializer(serializers.ModelSerializer):

    class Meta:
        model = Plan
        fields = '__all__'
        ordering = ['no_of_days']

class CurrentStatusSerializer(serializers.ModelSerializer):

    class Meta:
        model = CurrentStatus
        fields = '__all__'

    def to_representation(self, instance):
        response =  super().to_representation(instance)
        response['user'] = ProfileSerializer(instance.user).data
        response['plan'] = PlanSerializer(instance.plan).data
        return response
