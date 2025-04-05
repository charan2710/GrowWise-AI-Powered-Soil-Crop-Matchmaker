from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('plant-recommender/', views.plant_recommender, name='plant_recommender'),
]
