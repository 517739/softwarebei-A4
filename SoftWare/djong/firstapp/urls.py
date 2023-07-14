from django.urls import path

from . import views

urlpatterns = [
    path('', views.main, name='main'),
    path('phone', views.phone, name='phone'),
    path('id', views.id,name='id'),
    path('name', views.name,name='name'),
]