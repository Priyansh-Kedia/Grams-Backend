from rest_framework import serializers

from .models import Profile


class OTPSerializer(serializers.ModelSerializer):

    class Meta:
        model = Profile
        fields=['phone_number']
