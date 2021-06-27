from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
os.environ.setdefault('DJANGO_SETTINGS_MODULE','grams_backend.settings')
import django
django.setup()

from users.models import Profile
from trials.models import Plan,CurrentStatus
from .enums import TrialResponse

app = Celery('grams_backend')

# run command celery --app=grams_backend worker -l info -P gevent 
# namespace CELERY means that all celery related config keys will
# have CELERY_ as prefix
app.config_from_object('django.conf:settings', namespace='CELERY')

app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print('Request: {0!r}'.format(self.request))


@app.task
def basic(phone_number):
    profile = Profile.objects.get(phone_number=phone_number)
    current = CurrentStatus.objects.get(user = profile)
    plan = Plan.objects.get(name = TrialResponse.FREETRIAL2) 
    print('first trial done, start second trial with gst id')
    current.plan = plan
    current.save()

@app.task
def prompt_payment_renewal(phone_number):
    profile = Profile.objects.get(phone_number=phone_number)
    current = CurrentStatus.objects.get(user = profile)
    current.paid = False
    current.save()
    print('please renew your payment')

