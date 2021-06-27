from typing import no_type_check
from django.db import models
from django.db.models.base import Model
from users.models import Profile
# Create your models here.


class Plan(models.Model):
    name = models.CharField(max_length=40)
    price = models.IntegerField(blank=True,null=True)
    readings =models.IntegerField(null=True,blank=True)
    no_of_days = models.IntegerField(default=30)

def __str__(self):
    return self.name


class CurrentStatus(models.Model):
    start_date = models.DateTimeField(auto_now=True)
    end_date = models.DateTimeField(blank=True,null=True)
    paid = models.BooleanField(default=False)
    user = models.OneToOneField(Profile,on_delete=models.CASCADE,null=True,blank=True,related_name='user')   
    plan = models.ForeignKey(Plan,on_delete=models.CASCADE,null=True,blank=True,related_name='plan')


