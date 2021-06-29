from django.contrib import admin

from .models import Scan, SubType, Type

admin.site.register(Scan)
admin.site.register(Type)
admin.site.register(SubType)
