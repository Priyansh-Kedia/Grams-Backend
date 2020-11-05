from rest_framework import serializers
from django.core.validators import RegexValidator

from .models import Profile, Address

class OTPSerializer(serializers.Serializer):
    phone_regex = RegexValidator(regex = r'^\+\d{4,15}$', message = "Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed.")
    phone_number = serializers.CharField(max_length = 17, validators = [phone_regex])

class AddressSerializer(serializers.ModelSerializer):
    class Meta:
        model = Address
        fields = ['address', 'city', 'state', 'country', 'profile']

    def create(self):
        address = Address(
            profile = self.validated_data['profile'],
            address = self.validated_data['address'],
            city = self.validated_data['city'],
            state = self.validated_data['state'],
            country = self.validated_data['country']
        )
        address.save()

class ProfileSerializer(serializers.Serializer):
    phone_regex = RegexValidator(regex = r'^\+\d{4,15}$', message = "Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed.")
    phone_number = serializers.CharField(max_length = 17, validators = [phone_regex])
    company_name = serializers.CharField(max_length = 100, required = False)
    name = serializers.CharField(max_length = 100, required = False)
    designation = serializers.CharField(max_length = 100, required = False)
    email_id = serializers.CharField(max_length = 100, required = False)
