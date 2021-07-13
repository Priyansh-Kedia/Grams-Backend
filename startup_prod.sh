#!/bin/bash

waitress-serve --listen=*:8000 grams_backend.wsgi:application & celery -A grams_backend worker -l INFO & celery -A grams_backend beat