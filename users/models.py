from django.db import models
from django.core.validators import RegexValidator


phone_regex = RegexValidator(regex = r'^\+\d{4,15}$', message = "Phone number must be entered in the format: '+999999999'. Up to 15 digits allowed.")
otp_regex = RegexValidator(regex = r'^\d{4}')

class Profile(models.Model):
    company_name = models.CharField(max_length = 200, verbose_name = "Company Name", null = True, blank = True)
    phone_number = models.CharField(max_length = 17, validators = [phone_regex], verbose_name = 'Phone No.', unique = True)
    name = models.CharField(max_length = 200, null = True, blank = True)
    designation = models.CharField(max_length = 100, null = True, blank = True)
    email_id = models.EmailField(null = True, blank = True)
    is_agreed = models.BooleanField(default = False, help_text = "Please Tick If you agree to the Terms and Conditions of the Contract", null = True, blank = True)
    otp = models.IntegerField(validators = [otp_regex], null = True, blank = True)
    otp_timestamp = models.DateTimeField(auto_now = True, verbose_name = "OTP Created On")
    profile_id = models.AutoField(primary_key=True, name='profile_id')

    def __str__(self):
        return self.phone_number

    @classmethod
    def getAllAddresses(cls, id):
        profile = cls(pk = id)
        return profile.address_set.all()

    @classmethod
    def getAllScans(cls, phone_number):
        profile = cls.objects.get(phone_number = phone_number)
        # print(profile)
        # print(profile.scan_set.all())
        return profile.scan_set.all() 



    class Meta:
        verbose_name_plural = "Profiles"

class Address(models.Model):
    profile_id = models.ForeignKey(to = Profile, on_delete = models.CASCADE)
    address_id = models.AutoField(primary_key=True, verbose_name='address_id')
    address = models.CharField(max_length = 100, null = True, blank = True)
    city = models.CharField(max_length = 100, null = True, blank = True)
    state = models.CharField(max_length = 100, null = True, blank = True)
    country = models.CharField(max_length = 100, null = True, blank = True)

    def __str__(self):
        return self.address
        
    class Meta:
        verbose_name_plural = "Addresses"

class Image(models.Model):
    image = models.ImageField(upload_to = 'media/')
    name = models.CharField(max_length=50, default='QWERTY')
    def __str__(self):
        return str(self.id)