from rest_framework import serializers

from .models import Grain

class GrainSerializer(serializers.HyperlinkedModelSerializer):

    #image_url = serializers.SerializerMethodField()
    #image_url = serializers.ImageField(max_length=None, use_url=True, allow_null=True, required=False)
    #file_url = serializers.FileField(max_length=None, use_url=True, allow_null=True, required=False)
    class Meta:
        
        model = Grain
        fields = ('profile', 'image', 'csv_file')

    