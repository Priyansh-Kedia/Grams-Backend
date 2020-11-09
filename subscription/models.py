from django.db import models
from django.utils import timezone

from users.models import Profile

class Subscription(models.Model):
    profile = models.ForeignKey(to = Profile, on_delete = models.CASCADE)
    is_subscribed = models.BooleanField(null = True, blank = True)
    expiry_date = models.DateTimeField(null = True, blank = True)

    def __str__(self):
        return self.profile.phone_number
    
class Plan(models.Model):
    no_of_days = models.PositiveIntegerField(null = True, blank = True)
    price = models.PositiveIntegerField(null = True, blank = True)

    def __str__(self):
        return str(self.id)
   
