from django.urls import path, include
from watchlist_app.api import views
from rest_framework.routers import DefaultRouter


router = DefaultRouter()
router.register('platforms', views.StreamPlatformVS, basename="streamplatform")

urlpatterns = [
    path('', views.WatchListAV.as_view(), name='movie_list'),
    path('<int:pk>/', views.WatchListDetailsAV.as_view(), name='movie_details'),

    path('', include(router.urls)),
    
    path('<int:pk>/revs/create/', views.ReviewCreate.as_view(), name='review_create'),
    path('<int:pk>/revs/', views.ReviewList.as_view(), name='reviews_list'),
    path('revs/<int:pk>/', views.ReviewDetails.as_view(), name='review_details'),
    path('user-revs/', views.ReviewPerUser.as_view(), name='review_user'),

    
]
