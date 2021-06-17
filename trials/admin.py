from django.contrib import admin
from django.db.models.expressions import F
from .models import FreeTrial,Paid
# Register your models here.


admin.site.register(FreeTrial)
admin.site.register(Paid)
