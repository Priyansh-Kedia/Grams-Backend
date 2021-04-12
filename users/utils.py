from . import models
# from process_grains.models import Scan

import uuid
# DONT_USE = ['remove','approve','admin','teach',]

# def random_string_generator(size=10,chars=string.ascii_lowercase + string.digits):
#     return ''.join(random.choice(chars) for _ in range(size))

def unique_id_generator(instance,new_slug=None):
    if new_slug is not None:
        slug = new_slug
    else:
        slug = uuid.uuid4().hex[:16]
        # if instance.__class__ == models.Blog:
        #     slug = slugify(instance.title)
        # elif instance.__class__ == models.Topic:
        #     slug = slugify(instance.title)
        #     print(slug)
        # elif instance.__class__ == models.SubTopic:
        #     slug = slugify(instance.title)
        #     print(slug)
    # if slug in DONT_USE:
    #     new_slug = slug + random_string_generator(size=4)
    #     return unique_slug_generator(instance,new_slug=new_slug)

    Klass = instance.__class__
    qs_exists = Klass.objects.filter(scan_id=slug).exists()
    if qs_exists:
        new_slug = uuid.uuid1().hex
        return unique_id_generator(instance, new_slug=new_slug)
        # new_slug = "{slug}-{randstr}".format(slug=slug,randstr=random_string_generator(size=4))
        # return unique_slug_generator(instance, new_slug=new_slug)
    return slug
