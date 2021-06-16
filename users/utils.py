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

def run_ml_code(phone_number):
    ml_list, _ = main('onion1.jpg',20,0.25)
        # ml_data =  {
        # 'item_type' : "hello",
        # 'sub_type' : "hello",
        # 'created_on' : datetime.now(),
        # 'no_of_particles' : ml_list[0],
        # 'avg_area' : round(ml_list[1], 2),
        # 'avg_length' : round(ml_list[2], 2),
        # 'avg_width' : round(ml_list[3], 2),
        # 'avg_l_by_w' : round(ml_list[4], 2),
        # 'avg_circularity' : round(ml_list[5], 2),
        # 'lot_no' : "hello",
        # 'no_of_kernels' : ml_list[0],
        # }
        # ml_data['user'] = profile.pk

        # print(ml_data)
        # scan_serializer = ScanSerializer(data = ml_data)

        

        # if not scan_serializer.is_valid():
        #     return Response(scan_serializer.errors)

        
        
        # scan_serializer.save()
    heading_msg = "Your results of reading ID is available, View your result in the app"
    content_msg = "Your Reading has been successfully computed."
    data = {    "app_id": "fad6e42a-0b02-45d6-9ab0-a654b204aca9", "contents": {"en": content_msg}, "headings": {"en": heading_msg}, "include_external_user_ids": [phone_number] , "chrome_web_image": "https://images.ctfassets.net/hrltx12pl8hq/7yQR5uJhwEkRfjwMFJ7bUK/dc52a0913e8ff8b5c276177890eb0129/offset_comp_772626-opt.jpg?fit=fill&w=800&h=300"}

    requests.post(    "https://onesignal.com/api/v1/notifications",    headers={"Authorization": "Basic NDJkOGMyZDQtMjgyYi00Y2JkLWFjZTgtZGQ2NjQ1NDUwNzg3"}, json=data)
    print(ml_list)
