from django.db import models

from users.models import Profile 

class Scan(models.Model):
    user = models.ForeignKey(Profile, on_delete = models.CASCADE)
    scan_id = models.CharField(max_length = 100, null = True, blank = True)
    image = models.ImageField(upload_to = 'media/', null = True, blank =True)#image will be a list field
    item_type = models.CharField(max_length = 100, null = True, blank = True)
    sub_type = models.CharField(max_length = 100, null = True, blank = True)
    created_on = models.DateTimeField(auto_now_add = True)

    def __str__(self):
        return str(self.id)

# Create your models here.
