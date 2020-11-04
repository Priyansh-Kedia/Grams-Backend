from django.contrib import admin
from .models import Profile, Address

class ProfileAdmin(admin.ModelAdmin):
    readonly_fields = ('otp_timestamp', )


admin.site.register(Address)
admin.site.register(Profile, ProfileAdmin)
