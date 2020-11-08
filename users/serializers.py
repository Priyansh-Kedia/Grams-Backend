from rest_framework import serializers
from django.core.validators import RegexValidator

from .models import Profile, Address

class OTPSerializer(serializers.Serializer):
    phone_regex = RegexValidator(regex = r'^\+\d{4,15}$', message = "Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed.")
    phone_number = serializers.CharField(max_length = 17, validators = [phone_regex])

class AddressSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = Address
        fields = '__all__'
        

    def create(self):
        address = Address(
            profile = self.validated_data["profile"],
            address = self.validated_data["address"],
            city = self.validated_data["city"],
            state = self.validated_data["state"],
            country = self.validated_data["country"],
        )
        address.save()

    def update(self, instance):
        instance.address = self.validated_data.get('address', instance.address)
        instance.city = self.validated_data.get('city', instance.city)
        instance.state = self.validated_data.get('state', instance.state)
        instance.country = self.validated_data.get('country', instance.country)
        instance.save()
        return instance

class ProfileSerializer(serializers.Serializer):
    phone_regex = RegexValidator(regex = r'^\+\d{4,15}$', message = "Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed.")
    phone_number = serializers.CharField(max_length = 17, validators = [phone_regex],required = False)
    company_name = serializers.CharField(max_length = 100, required = False)
    name = serializers.CharField(max_length = 100, required = False)
    designation = serializers.CharField(max_length = 100, required = False)
    email_id = serializers.EmailField(max_length = 100, required = False)
    is_agreed = serializers.BooleanField(required = False)

    def update(self, instance):
        #phone_number = self.validated_data.get('phone_number')
        instance.phone_number = self.validated_data.get('phone_number', instance.phone_number)
        instance.company_name = self.validated_data.get('company_name', instance.company_name)
        instance.name = self.validated_data.get('name', instance.name)
        instance.designation = self.validated_data.get('designation', instance.designation)
        instance.email_id = self.validated_data.get('email_id', instance.email_id)
        instance.is_agreed = self.validated_data.get('is_agreed', instance.is_agreed)
        instance.save()
        return instance
