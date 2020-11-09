from rest_framework import serializers

from .models import Grain

class GrainSerializer(serializers.ModelSerializer):

    
    class Meta:
        
        model = Grain
        fields = ('profile', 'image')

    