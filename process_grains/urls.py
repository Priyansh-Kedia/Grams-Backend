from django.urls import path

from . import views

urlpatterns=[
    path('', views.retreive_scan, name = 'retrieve_scan'),
]