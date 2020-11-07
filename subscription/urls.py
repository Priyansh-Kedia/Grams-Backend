from django.urls import path
from . import views

urlpatterns =[
    path('my_plans/', views.my_plans, name = 'retrieve'),
    path('add/', views.add_subscription, name = 'add_subscription'),
]