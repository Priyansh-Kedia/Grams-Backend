from rest_framework import serializers
from django.core.validators import RegexValidator
import re
from .models import Profile, Address

class OTPSerializer(serializers.Serializer):
    phone_regex = RegexValidator(regex = r'^\+\d{4,15}$', message = "Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed.")
    phone_number = serializers.CharField(max_length = 17, validators = [phone_regex])

class AddressSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = Address
        fields = '__all__'
    
    def first_letter_capitalized_form(self, field_value):
        field_value = " ".join(w.capitalize() for w in re.split('\\s+', field_value))
        return field_value

    def lowercase_form(self, field_value):
        field_value = field_value.lower()
        return field_value
    
    def create(self):
        address = Address(
            profile = self.validated_data["profile"],
            address = self.first_letter_capitalized_form(self.validated_data["address"]),
            city = self.first_letter_capitalized_form(self.validated_data["city"]),
            state = self.first_letter_capitalized_form(self.validated_data["state"]),
            country = self.first_letter_capitalized_form(self.validated_data["country"]),
        )
        address.save()
        return address

    def update(self, instance):
        instance.address = self.first_letter_capitalized_form(self.validated_data.get('address', instance.address))
        instance.city = self.first_letter_capitalized_form(self.validated_data.get('city', instance.city))
        instance.state = self.first_letter_capitalized_form(self.validated_data.get('state', instance.state))
        instance.country = self.first_letter_capitalized_form(self.validated_data.get('country', instance.country))
        instance.save()
        return instance

class ProfileSerializer(serializers.ModelSerializer):
    phone_regex = RegexValidator(regex = r'^\+\d{4,15}$', message = "Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed.")
    phone_number = serializers.CharField(max_length = 17, validators = [phone_regex],required = False)
    company_name = serializers.CharField(max_length = 100, required = False,allow_null = True, allow_blank = True)
    name = serializers.CharField(max_length = 100, required = False,allow_null = True, allow_blank = True)
    designation = serializers.CharField(max_length = 100, required = False,allow_null = True, allow_blank = True)
    email_id = serializers.EmailField(max_length = 100, required = False,allow_null = True, allow_blank = True)
    is_agreed = serializers.BooleanField(required = False)

    class Meta:
        model = Profile
        fields = ['name', 'company_name', 'name', 'email_id', 'is_agreed', 'phone_number', 'designation']

    def first_letter_capitalized_form(self, field_value):
        field_value = " ".join(w.capitalize() for w in re.split('\\s+', field_value))
        return field_value

    def lowercase_form(self, field_value):
        field_value = field_value.lower()
        return field_value

    def update(self, instance):       
        instance.phone_number = self.validated_data.get('phone_number', instance.phone_number)
        instance.company_name = self.first_letter_capitalized_form(self.validated_data.get('company_name', instance.company_name))
        instance.name = self.first_letter_capitalized_form(self.validated_data.get('name', instance.name))
        instance.designation = self.first_letter_capitalized_form(self.validated_data.get('designation', instance.designation))
        instance.email_id = self.lowercase_form(self.validated_data.get('email_id', instance.email_id))
        instance.is_agreed = self.validated_data.get('is_agreed', instance.is_agreed)
        instance.save()
        return instance
