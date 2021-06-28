from django.urls import path

from . import views

urlpatterns=[
    path('retrieve_scans/', views.retreive_scan, name = 'retrieve_scan'),
    path('delete_reading/', views.delete_readings, name = 'delete_reading'),
    path('get_all_types/', views.get_all_types, name = 'get_all_types'),
    path('get_subtype/', views.get_subtype, name = 'get_subtype')
]