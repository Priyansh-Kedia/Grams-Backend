#!/bin/bash

python manage.py runserver 0.0.0.0:8000 & celery -A grams_backend worker -l INFO & celery -A grams_backend beat