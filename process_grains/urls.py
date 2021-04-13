from django.urls import path

from . import views

urlpatterns=[
    path('retrieve_scans/', views.retreive_scan, name = 'retrieve_scan'),
    path('delete_reading/', views.delete_readings, name = 'delete_reading')
]