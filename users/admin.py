from django.contrib import admin
from .models import Feedback, Profile, Address, Image

class ProfileAdmin(admin.ModelAdmin):
    readonly_fields = ('otp_timestamp', )


admin.site.register(Address)
admin.site.register(Profile, ProfileAdmin)
admin.site.register(Image)
admin.site.register(Feedback)