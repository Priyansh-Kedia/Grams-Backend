from django.urls import path
from . import views

urlpatterns=[

    path('generate/', views.generate_otp, name = 'generate_otp'),
    path('verify/', views.verify_otp, name = 'verify_otp'),
    path('update/', views.update, name = 'update'),
    path('add_address/', views.add_address, name = 'add_address'),
    path('retrieve/', views.retrieve_address, name = 'retrieve_address'),

]
