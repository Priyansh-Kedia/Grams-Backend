from django.urls import path
from . import views

urlpatterns=[

    path('generate/', views.generate_otp, name = 'generate_otp'),
    path('verify/', views.verify_otp, name = 'verify_otp'),
    path('update_profile/', views.update_profile, name = 'update_profile'),
    path('update_address/', views.update_address, name = 'update_address'),
    path('add_address/', views.add_address, name = 'add_address'),
    path('retrieve_address/', views.retrieve_address, name = 'retrieve_address'),
    path('retrieve_profile', views.retrieve_profile, name = 'retrieve_profile'),
    path('upload_image/<str:phone_number>/<str:type>/<str:sub_type>/', views.upload_image),
    path('health/', views.health, name = 'health'),
    path('upload/', views.MyImageView.as_view()),
    path('feedback/', views.feedback, name = 'feedback'),

]
