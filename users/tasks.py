from __future__ import absolute_import, unicode_literals

from matplotlib.pyplot import sca
from grams_backend import Constants
from django.core.files import File

from celery import shared_task

from .models import Image, Profile, Address
from trials.models import CurrentStatus
from process_grains.serializers import ScanSerializer
from main import main
import requests
import datetime
import os
from decouple import config

@shared_task
def add(x, y):
    profiles = Address.objects.filter(city='Jaipur')
    profiles.delete()
    return x+y

@shared_task
def run_ml_code(phone_number,image_url,item_type,sub_type, image_obj):
    profile = Profile.objects.get(phone_number=phone_number)
    image_url = os.getcwd() + image_url
    try:
        ml_list, _, csv_name = main(image_url)
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
        'no_of_kernels' : ml_list[0]
        }
        ml_data['user'] = profile.profile_id
        scan_serializer = ScanSerializer(data = ml_data)
        if scan_serializer.is_valid():
            print("valid")
        scan = scan_serializer.save()
        scan.image = Image.objects.get(id = image_obj)
        print(csv_name)
        scan.output_csv = File(open(csv_name, 'rb'))
        scan.save()
        ml_data["image"] = image_url
        ml_data["scan_id"] = scan.scan_id
        heading_msg = "Your results of reading ID {} is available, View your result in the app".format(scan.scan_id)
        content_msg = "Your Reading has been successfully computed."
        data = { "app_id": config(Constants.APP_ID), "contents": {"en": content_msg}, "headings": {"en": heading_msg}, "include_external_user_ids": [phone_number] , "chrome_web_image": config(Constants.CHROME_WEB_IMAGE)}
        requests.post("https://onesignal.com/api/v1/notifications",    headers={"Authorization": "Basic " +  config(Constants.API_KEY)}, json=data)
    except Exception as e:
        current = CurrentStatus.objects.get(user = profile)
        current.no_of_readings += 1
        current.save()
        heading_msg = "Your results are unable to be computed"
        content_msg = str(e)
        data = {
            "app_id": config(Constants.APP_ID), 
            "contents": {"en": content_msg}, 
            "headings": {"en": heading_msg}, 
            "include_external_user_ids": [phone_number] , 
            "chrome_web_image": config(Constants.CHROME_WEB_IMAGE)
        }
        requests.post("https://onesignal.com/api/v1/notifications",headers={"Authorization": "Basic " +  config(Constants.API_KEY)}, json=data)
