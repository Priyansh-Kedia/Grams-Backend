from django.urls import path

from . import views

urlpatterns=[
    path('retrieve_scans/', views.retreive_scan, name = 'retrieve_scan'),
]