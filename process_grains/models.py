from django.db import models

from users.models import Profile 

class Scan(models.Model):
    user = models.ForeignKey(Profile, on_delete = models.CASCADE)
    scan_id = models.CharField(max_length = 100, null = True, blank = True)
    image = models.ImageField(upload_to = 'media/', null = True, blank =True)
    item_type = models.CharField(max_length = 100, null = True, blank = True)
    sub_type = models.CharField(max_length = 100, null = True, blank = True)
    created_on = models.DateTimeField(auto_now_add = True)
    no_of_particles = models.IntegerField(null = True, blank = True)
    avg_area = models.DecimalField(max_digits = 5, decimal_places = 2, null = True, blank = True)
    avg_length = models.DecimalField(max_digits = 5, decimal_places = 2, null = True, blank = True)
    avg_width = models.DecimalField(max_digits = 5, decimal_places = 2, null = True, blank = True)
    avg_l_by_w = models.DecimalField(max_digits = 5, decimal_places = 2, null = True, blank = True)
    avg_circularity = models.DecimalField(max_digits = 5, decimal_places = 2, null = True, blank = True)
    lot_no = models.CharField(max_length = 100, null = True, blank = True)
    no_of_kernels = models.IntegerField(null = True, blank = True)

    def __str__(self):
        return str(self.id)

    @classmethod
    def getPhoneNumberByUser(cls, user):
        phone_number = cls.objects.filter(user = user)[0].user.phone_number
        return phone_number

# Create your models here.
