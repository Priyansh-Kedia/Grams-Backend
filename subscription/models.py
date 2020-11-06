from django.db import models

from users.models import Profile

class Subscription(models.Model):
    profile = models.ForeignKey(to = Profile, on_delete = models.CASCADE)
    is_subscribed = models.BooleanField()
    days_left = models.PositiveIntegerField(null = True, blank = True)

# Create your models here.
