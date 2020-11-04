from django.db import models
from django.core.validators import RegexValidator
#from django_countries.fields import CountryField


class Address(models.Model):
    #profile=models.ForeignKey(Profile,on_delete=models.CASCADE)
    address = models.CharField(max_length=100)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    country = models.CharField(max_length=100)

    def __str__(self):
        return self.address

class Profile(models.Model):
    phone_regex = RegexValidator(regex=r'^\+?1?\d{4,15}$',message="Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed.")
    company_name = models.CharField(max_length=200,verbose_name="Company Name")
    contact_person_details = models.CharField(max_length=17,validators=[phone_regex],verbose_name='Phone No.')
    name = models.CharField(max_length=200)
    address = models.ManyToManyField(Address)
    designation = models.CharField(max_length=100)
    email_id = models.EmailField()
    i_agree = models.BooleanField(default=False,required=True,help_text="Please Tick If you agree to the Terms and Conditions of the Contract")

    def __str__(self):
        return self.name


# Create your models here.
