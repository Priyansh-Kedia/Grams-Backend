from rest_framework import serializers
from .models import Plan,Paid

class PlanSerializer(serializers.ModelSerializer):

    class Meta:
        model = Plan
        fields = '__all__'


class PaidSerializer(serializers.ModelSerializer):

    class Meta:
        model = Paid
        fields = '__all__'        