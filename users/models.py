from django.db import models
from django.core.validators import RegexValidator
from django.utils import timezone
#from django_countries.fields import CountryField




class Profile(models.Model):

    otp_regex = RegexValidator(regex = r'^\d{4}')
    company_name = models.CharField(max_length = 200, verbose_name = "Company Name",null = True, blank = True)
    phone_number = models.CharField(max_length = 17 , verbose_name = 'Phone No.', null = True, blank = True)
    name = models.CharField(max_length = 200, null = True, blank = True)
    designation = models.CharField(max_length = 100,null = True, blank = True)
    email_id = models.EmailField(null = True,blank = True)
    is_agreed = models.BooleanField(default = False, help_text = "Please Tick If you agree to the Terms and Conditions of the Contract",null = True,blank = True)
    otp = models.IntegerField(validators = [otp_regex])
    otp_timestamp = models.DateTimeField(auto_now = True)

    def __str__(self):
        return self.name


class Address(models.Model):
    #profile=models.ForeignKey(Profile,on_delete=models.CASCADE)
    profile = models.ForeignKey(to = Profile, on_delete = models.CASCADE, null = True, blank = True)
    address = models.CharField(max_length = 100, null = True, blank = True)
    city = models.CharField(max_length = 100, null = True, blank = True)
    state = models.CharField(max_length = 100, null = True, blank = True)
    country = models.CharField(max_length = 100, null = True, blank = True)

    def __str__(self):
        return self.address


# Create your models here.
