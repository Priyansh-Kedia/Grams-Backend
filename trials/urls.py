from django.urls import path
from . import views

urlpatterns=[
    path('plan_status/<str:phone_number>', views.plan_status, name = 'plan_status'),
    path('get_all_plans/', views.get_all_plans, name = 'get_all_plans'),
    path('update_payment/', views.update_payment_status, name = 'update_payment'),
]