from django.db.models import fields
from rest_framework import serializers

from .models import Scan, Type, SubType

class ScanSerializer(serializers.ModelSerializer):

    class Meta:
        model = Scan
        fields = '__all__'

class TypeSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = Type
        fields = '__all__'

class SubTypeSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = SubType
        fields = '__all__'
    
    def to_representation(self, instance):
        response =  super().to_representation(instance)
        response['type'] = TypeSerializer(instance.type).data
        return response


