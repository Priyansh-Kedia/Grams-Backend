from django.db import models
from django.db.models.base import Model

# Create your models here.
class FreeTrial(models.Model):
    t1 = models.IntegerField(blank=True,null=True)
    r1 = models.IntegerField(blank=True,null=True)
    t2 = models.IntegerField(blank=True,null=True)    
    r2 = models.IntegerField(blank=True,null=True)    
    first_trial = models.BooleanField(default=False)
    second_trial = models.BooleanField(default=False)
    start_date = models.DateTimeField(auto_now_add=True)
    end_date = models.DateTimeField(blank=True,null=True)    


class Paid(models.Model):
    p1 = models.IntegerField(blank=True,null=True)
    d1 = models.IntegerField(blank=True,null=True)
    p2 = models.IntegerField(blank=True,null=True)    
    d2 = models.IntegerField(blank=True,null=True) 
    start_date = models.DateTimeField(auto_now_add=True)
    end_date = models.DateTimeField(blank=True,null=True)
    paid = models.BooleanField(default=False)