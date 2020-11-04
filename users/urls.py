from django.urls import path
from . import views

urlpatterns=[

    path('generate/', views.generate_otp, name = 'generate_otp'),
    path('verify/', views.verify_otp, name = 'verify_otp'),

]
