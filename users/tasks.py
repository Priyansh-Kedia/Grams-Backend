from __future__ import absolute_import, unicode_literals

from celery import shared_task

from .models import Profile, Address
from process_grains.serializers import ScanSerializer
from main import main
import requests
import datetime
import os

@shared_task
def add(x, y):
    profiles = Address.objects.filter(city='Jaipur')
    profiles.delete()
    return x+y


@shared_task
def run_ml_code(phone_number,image_url,item_type,sub_type):
    profile = Profile.objects.get(phone_number=phone_number)
    image_url = os.getcwd() + image_url
    ml_list, _ = main(image_url,20,0.25)
    ml_data =  {
    'item_type' : item_type,
    'sub_type' : sub_type,
    'created_on' : datetime.datetime.now(),
    'no_of_particles' : ml_list[0],
    'avg_area' : round(ml_list[1], 2),
    'avg_length' : round(ml_list[2], 2),
    'avg_width' : round(ml_list[3], 2),
    'avg_l_by_w' : round(ml_list[4], 2),
    'avg_circularity' : round(ml_list[5], 2),
    'lot_no' : "hello",
    'no_of_kernels' : ml_list[0],
    }
    ml_data['user'] = profile.pk
    print(ml_data)
    scan_serializer = ScanSerializer(data = ml_data)
    if not scan_serializer.is_valid():
        print(scan_serializer.errors)    
    scan_serializer.save()
    heading_msg = "Your results of reading ID is available, View your result in the app"
    content_msg = "Your Reading has been successfully computed."
    data = { "app_id": "fad6e42a-0b02-45d6-9ab0-a654b204aca9", "contents": {"en": content_msg}, "headings": {"en": heading_msg}, "include_external_user_ids": [phone_number] , "chrome_web_image": "https://images.ctfassets.net/hrltx12pl8hq/7yQR5uJhwEkRfjwMFJ7bUK/dc52a0913e8ff8b5c276177890eb0129/offset_comp_772626-opt.jpg?fit=fill&w=800&h=300"}

    requests.post(    "https://onesignal.com/api/v1/notifications",    headers={"Authorization": "Basic NDJkOGMyZDQtMjgyYi00Y2JkLWFjZTgtZGQ2NjQ1NDUwNzg3"}, json=data)
    print(ml_list)
