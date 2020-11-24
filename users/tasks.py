from __future__ import absolute_import, unicode_literals

from celery import shared_task

from .models import Profile, Address

@shared_task
def add(x, y):
    profiles = Address.objects.filter(city='Jaipur')
    profiles.delete()
    return x+y