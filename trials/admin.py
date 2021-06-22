from django.contrib import admin
from django.db.models.expressions import F
from .models import FreeTrial,Paid, Plan
# Register your models here.


admin.site.register(FreeTrial)
admin.site.register(Paid)
admin.site.register(Plan)