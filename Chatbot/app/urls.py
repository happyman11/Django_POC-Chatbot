from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('upload/', views.upload_file),
    path('chat/', views.chat),
    path('delete_session/', views.delete_session),
     
]