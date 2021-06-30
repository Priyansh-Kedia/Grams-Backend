from rest_framework import serializers
from django.core.validators import RegexValidator
import re
from .models import Profile, Address, Image

class OTPSerializer(serializers.Serializer):
    phone_regex = RegexValidator(regex = r'^\+\d{4,15}$', message = "Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed.")
    phone_number = serializers.CharField(max_length = 17, validators = [phone_regex])

class AddressSerializer(serializers.ModelSerializer):
    address_id = serializers.PrimaryKeyRelatedField(read_only=True)
    profile_id = serializers.PrimaryKeyRelatedField(queryset = Profile.objects.all(),required = False)  

    class Meta:
        model = Address
        fields = ['address', 'city', 'country', 'state', 'profile_id', 'address_id']
        extra_kwargs = {
            'address_id': {'read_only': True}
        }
 
    def first_letter_capitalized_form(self, field_value):
        if field_value is not None:
            field_value = " ".join(w.capitalize() for w in re.split('\\s+', field_value))
            return field_value
        return field_value 

    def lowercase_form(self, field_value):
        if field_value is not None:
            field_value = field_value.lower()
            return field_value
        return field_value
    
    def create(self):
        address = Address(
            profile_id = self.validated_data['profile_id'],
            address = self.first_letter_capitalized_form(self.validated_data.get("address", None)),
            city = self.first_letter_capitalized_form(self.validated_data.get("city", None)),
            state = self.first_letter_capitalized_form(self.validated_data.get("state", None)),
            country = self.first_letter_capitalized_form(self.validated_data.get("country", None)),
            
        )
        address.save()
        return address

    def update(self, instance):
        instance.address = self.first_letter_capitalized_form( (self.validated_data['address'],instance.address)[self.validated_data.get('address') is None])
        instance.city = self.first_letter_capitalized_form((self.validated_data['city'],instance.city)[self.validated_data.get('city') is None])
        instance.state = self.first_letter_capitalized_form((self.validated_data['state'],instance.state)[self.validated_data.get('state') is None])
        instance.country = self.first_letter_capitalized_form((self.validated_data['country'],instance.country)[self.validated_data.get('country') is None])
        instance.save()
        return instance

class ProfileSerializer(serializers.ModelSerializer):
    phone_regex = RegexValidator(regex = r'^\+\d{4,15}$', message = "Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed.")
    phone_number = serializers.CharField(max_length = 17, validators = [phone_regex], required = False, allow_null = True)
    
    class Meta:
        model = Profile
        fields = [ 'name', 'company_name', 'email_id', 'is_agreed', 'phone_number', 'designation','gst_no']
        extra_kwargs = {
            'profile_id': {'read_only': True}
        }

    def first_letter_capitalized_form(self, field_value):
        if field_value is not None:
            field_value = " ".join(w.capitalize() for w in re.split('\\s+', field_value))
            return field_value
        return field_value

    def lowercase_form(self, field_value):
        if field_value is not None:
            field_value = field_value.lower()
            return field_value
        return field_value

    def update(self, instance):       
        instance.phone_number = (self.validated_data['phone_number'],instance.phone_number)[self.validated_data.get('phone_number') is None]
        instance.company_name = self.first_letter_capitalized_form( (self.validated_data['company_name'],instance.company_name)[self.validated_data.get('company_name') is None])
        instance.name = self.first_letter_capitalized_form((self.validated_data['name'],instance.name)[self.validated_data.get('name') is None])
        instance.designation = self.first_letter_capitalized_form((self.validated_data['designation'],instance.designation)[self.validated_data.get('designation') is None])
        instance.email_id = self.lowercase_form((self.validated_data['email_id'],instance.email_id)[self.validated_data.get('email_id') is None])
        instance.is_agreed = self.validated_data.get('is_agreed', instance.is_agreed)
        instance.save()
        return instance

class ImageSerializer(serializers.ModelSerializer):

    class Meta:
        model = Image
        fields = '__all__'