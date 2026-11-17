from django.contrib.auth.models import User
from django.urls import reverse

from rest_framework import status
from rest_framework.test import APITestCase
from rest_framework.authtoken.models import Token

from watchlist_app.api import serializers
from watchlist_app import models

# class StreamPlatformTestCase(APITestCase):

#     def setUp(self):
#         self.user = User.objects.create_user(username='Example', password='Pass-@26') # user created
#         self.token = Token.objects.get(user__username=self.user) # token fetched (created prev. step)
#         self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token.key) # token used in authorization
#         self.stream = models.StreamPlatform.objects.create(name='Netflix', about='Top 5 platforms',
#                                                            website='https://netflix.com')

#     def test_create_streamplatform(self):
#         data = {
#             'name' : 'Netflix',
#             'about' : 'Number one platform !',
#             'website' : 'http://netflix.com'
#         }

#         response = self.client.post(reverse('streamplatform-list'), data)
#         self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)


#     def test_list_streamplatform(self):
#         response = self.client.get(reverse('streamplatform-list'))
#         self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    
#     def test_ret_streamplatform(self):
#         response = self.client.get(reverse('streamplatform-detail', args=(self.stream.id,)))
#         self.assertEqual(response.status_code, status.HTTP_200_OK)


#     def test_destroy_streamplatform(self):
#         response = self.client.delete(reverse('streamplatform-detail', args=(self.stream.id,)))
#         self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)


#     def test_update_streamplatform(self):
#         data = {
#             'name' : 'Netflix',
#             'about' : 'Number one platform !',
#             'website' : 'http://netflix.com'
#         }
#         response = self.client.put(reverse('streamplatform-detail', args=(self.stream.id,)), data)
#         self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)


class WatchlistTestCase(APITestCase):

    def setUp(self):
        self.admin_user = User.objects.create_superuser(
            username='admin',
            password='password123',
            email='admin@example.com'
        )
        self.user = User.objects.create_user(username='Example', password='Pass-@26') # user created
        self.token = Token.objects.get(user__username=self.user) # token fetched (created prev. step)
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token.key) # token used in authorization
        self.stream = models.StreamPlatform.objects.create(name='Netflix', about='Top 5 platforms',
                                                               website='https://netflix.com')
        self.media = models.WatchList.objects.create(platform=self.stream, storyline='Asian princess',
                                                           title='Mulan 2', active=True)

    def test_create_watchlist_NA(self):
        data = {
            'platform' : self.stream,
            'title' : 'Mulan',
            'storyline' : 'Asian princess',
            'active' : True
        }
        response = self.client.post(reverse('movie_list'), data)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)


    def test_create_watchlist_admin(self):
        self.client.force_authenticate(user=self.admin_user)
        data = {
            'platform' : self.stream,
            'title' : 'Mulan',
            'storyline' : 'Asian princess',
            'active' : True
        }
        response = self.client.post(reverse('movie_list'), data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        if response.status_code != 201:
            print(response.data)


    def test_list_watchlist(self):
        response = self.client.get(reverse('movie_list'))
        self.assertEqual(response.status_code, status.HTTP_200_OK)


    def test_ret_watchlist(self):
        response = self.client.get(reverse('movie_details', args=(self.media.id,)))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(models.WatchList.objects.count(), 1)
        self.assertEqual(models.WatchList.objects.get().title, 'Mulan 2')


    def test_destroy_watchlist(self):
        response = self.client.delete(reverse('movie_details', args=(self.media.id,)))
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    
    def test_destroy_watchlist_admin(self):
        self.client.force_authenticate(user=self.admin_user)
        response = self.client.delete(reverse('movie_details', args=(self.media.id,)))
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)


    def test_update_watchlist(self):
        data = {
           'platform' : self.stream,
           'title' : 'Mulan',
           'storyline' : 'Asian princess',
           'active' : True
        }
        response = self.client.put(reverse('movie_details', args=(self.media.id,)), data)
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)


    def test_update_watchlist_admin(self):
        self.client.force_authenticate(user=self.admin_user)
        data = {
           'platform' : self.stream,
           'title' : 'Mulan',
           'storyline' : 'Asian princess',
           'active' : True
        }
        response = self.client.put(reverse('movie_details', args=(self.media.id,)), data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    

# class ReviewTestCase(APITestCase):

#     def setUp(self):
#         self.user = User.objects.create_user(username='Example', password='Pass-@26') # user created
#         self.token = Token.objects.get(user__username=self.user) # token fetched (created prev. step)
#         self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token.key) # token used in authorization
#         self.stream = models.StreamPlatform.objects.create(name='Netflix', about='Top 5 platforms',
#                                                                website='https://netflix.com')
#         self.media = models.WatchList.objects.create(platform=self.stream, storyline='Asian princess',
#                                                            title='Mulan 2', active=True)
#         self.media2 = models.WatchList.objects.create(platform=self.stream, storyline='Asian princess',
#                                                            title='Mulan 2', active=True)
#         self.review = models.Review.objects.create(reviewer=self.user, rating=4,
#                description='Brings about nice memories', active=True, watchlist=self.media)

#     def test_review_create(self):
#         data = {
#             'reviewer' : self.user.id,
#             'rating' : 5,
#             'description' : 'Brings about nice memories from childhood (=',
#             'watchlist' : self.media2.id,
#             'active' : True
#         }
#         response = self.client.post(reverse('review_create', args=(self.media2.id,)), data)
#         self.assertEqual(response.status_code, status.HTTP_201_CREATED)

#         response = self.client.post(reverse('review_create', args=(self.media2.id,)), data)
#         self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST) # second review


#     def test_review_create_unauth(self):
#         data = {
#             'reviewer' : self.user.id,
#             'rating' : 5,
#             'description' : 'Brings about nice memories from childhood (=',
#             'watchlist' : self.media.id,
#             'active' : True
#         }
#         self.client.force_authenticate(user=None) # logout
#         response = self.client.post(reverse('review_create', args=(self.media.id,)), data)
#         self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


#     def test_review_update(self):
#         data = {
#             'reviewer' : self.user.id,
#             'rating' : 5,
#             'description' : 'Brings about nice memories from childhood (=',
#             'watchlist' : self.media.id,
#             'active' : True
#         }
#         response = self.client.put(reverse('review_details', args=(self.review.id,)), data)
#         self.assertEqual(response.status_code, status.HTTP_200_OK)

#     def test_review_list(self):
#         response = self.client.get(reverse('reviews_list', args=(self.media.id,)))
#         self.assertEqual(response.status_code, status.HTTP_200_OK)
        
#     def test_review_ret(self):
#         response = self.client.get(reverse('review_details', args=(self.review.id,)))
#         self.assertEqual(response.status_code, status.HTTP_200_OK)
        
#     def test_review_destroy(self):
#         response = self.client.delete(reverse('review_details', args=(self.review.id,)))
#         print(response)
#         self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)

#     def test_reviewperuser(self):
#         response = self.client.get('/api/watch/user-revs/?username=' + self.user.username, args=(self.user.username,))
#         self.assertEqual(response.status_code, status.HTTP_200_OK)