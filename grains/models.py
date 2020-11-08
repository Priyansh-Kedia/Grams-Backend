from django.db import models

from users.models import Profile

class Grain(models.Model):
    profile = models.ForeignKey(to = Profile, on_delete = models.CASCADE)
    time = models.DateTimeField(auto_now_add= True)
    csv_file = models.FileField()
    image = models.ImageField(upload_to ='media/' )
