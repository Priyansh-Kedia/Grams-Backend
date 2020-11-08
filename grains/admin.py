from django.contrib import admin

from .models import Grain

class GrainAdmin(admin.ModelAdmin):
    readonly_fields = ('time', )
    list_display = ('profile','time','csv_file','image')

admin.site.register(Grain, GrainAdmin)
