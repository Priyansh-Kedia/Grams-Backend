from django.db import models
from django.db.models.base import Model
from users.models import Profile
# Create your models here.
class FreeTrial(models.Model):
    t1 = models.IntegerField(blank=True,null=True,default=15)
    r1 = models.IntegerField(blank=True,null=True,default=10)
    t2 = models.IntegerField(blank=True,null=True,default=15)    
    r2 = models.IntegerField(blank=True,null=True,default=10)    
    first_trial = models.BooleanField(default=False)
    second_trial = models.BooleanField(default=False)
    start_date = models.DateTimeField(auto_now_add=True)
    end_date = models.DateTimeField(blank=True,null=True) 
    user = models.OneToOneField(Profile,on_delete=models.CASCADE,null=True,blank=True)


class Plan(models.Model):
    name = models.CharField(max_length=40)



class Paid(models.Model):
    p1 = models.IntegerField(blank=True,null=True)
    d1 = models.IntegerField(blank=True,null=True)
    p2 = models.IntegerField(blank=True,null=True)    
    d2 = models.IntegerField(blank=True,null=True) 
    start_date = models.DateTimeField(auto_now_add=True)
    end_date = models.DateTimeField(blank=True,null=True)
    paid = models.BooleanField(default=False)
    user = models.OneToOneField(Profile,on_delete=models.CASCADE,null=True,blank=True)   
    plan = models.ForeignKey(Plan,on_delete=models.CASCADE,null=True,blank=True)


