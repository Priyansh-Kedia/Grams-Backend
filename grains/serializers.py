from rest_framework import serializers

from .models import Grain

class GrainSerializer(serializers.ModelSerializer):

    #image_url = serializers.SerializerMethodField()
    image_url = serializers.ImageField(max_length=None, use_url=True, allow_null=True, required=False)

    class Meta:
        
        model = Grain
        fields = ('profile', 'csv_file', 'image', 'image_url')

    