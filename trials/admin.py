from django.contrib import admin
from django.db.models.expressions import F
from .models import Plan,CurrentStatus
# Register your models here.

admin.site.register(Plan)
admin.site.register(CurrentStatus)