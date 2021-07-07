from grams_backend.Constants import PROFILE
from django.contrib import admin
from .models import Feedback, Profile, Address, Image
from trials.models import CurrentStatus

class CurrentStatusInline(admin.StackedInline):
    model = CurrentStatus
    extra = 0
    readonly_fields = ['start_date']

class ProfileAdmin(admin.ModelAdmin):
    model = Profile
    readonly_fields = ('otp_timestamp', )
    inlines = [CurrentStatusInline]


admin.site.register(Address)
admin.site.register(Profile, ProfileAdmin)
admin.site.register(Image)
admin.site.register(Feedback)