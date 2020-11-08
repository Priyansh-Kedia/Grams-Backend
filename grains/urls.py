from django.urls import path

from .models import Grain
from . import views

urlpatterns = [
    path('add/', views.add_grain, name = 'add_grain'),
]