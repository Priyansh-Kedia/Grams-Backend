from . import models
import random
import secrets
import uuid
import requests
from main import main

def unique_id_generator(instance,new_slug=None):
    if new_slug is not None:
        slug = new_slug
    else:
        slug = secrets.token_hex(8)


    Klass = instance.__class__
    qs_exists = Klass.objects.filter(scan_id=slug).exists()
    if qs_exists:
        new_slug = secrets.token_hex(8)
        return unique_id_generator(instance, new_slug=new_slug)
    return slug

def otp_generator():
    return random.randint(1111, 9999)

