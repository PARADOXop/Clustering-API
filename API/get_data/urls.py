from django.urls import path

from get_data import views

urlpatterns = [
    path("clusters/", views.clusters)
]
