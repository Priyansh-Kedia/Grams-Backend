from django.urls import path
from . import views

urlpatterns =[
    path('', views.add_subscription, name = 'add_subscription'),
]