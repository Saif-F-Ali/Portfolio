from django.shortcuts import get_object_or_404
from django_filters.rest_framework import DjangoFilterBackend

from rest_framework import filters, generics, mixins, status, viewsets
from rest_framework.exceptions import ValidationError
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from watchlist_app.api.pagination import CustomPaginationPNP
from watchlist_app.api.permissions import AdminOrReadOnly, ReviewerOrReadOnly
from watchlist_app.api.serializers import (
    ReviewSerializer,
    StreamPlatformSerializer,
    WatchListSerializer,
)
from watchlist_app.models import Review, StreamPlatform, WatchList



class WatchListAV(APIView):
    permission_classes = [AdminOrReadOnly]

    def get(self, request):  
        movies = WatchList.objects.all()
        paginator = CustomPaginationPNP()
        paginated_queryset = paginator.paginate_queryset(movies, request, view=self)
        serializer = WatchListSerializer(instance=paginated_queryset, many=True)
        return paginator.get_paginated_response(serializer.data)
    

    def post(self, request):
        title = request.data.get('title')
        warning_msg = None
        if title and WatchList.objects.filter(title=title).exists():
            warning_msg = f"Note: A movie named '{title}' already exists."
        
        serializer = WatchListSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()

            # adding context
            response_serializer = WatchListSerializer(
                serializer.instance, 
                context={'warning_message': warning_msg}
            )
            return Response(response_serializer.data)
        else:
            return Response(serializer.errors)


class WatchListDetailsAV(APIView):
    permission_classes = [AdminOrReadOnly]

    def get(self, request, pk):
        try:
            movie = WatchList.objects.get(pk=pk)
        except WatchList.DoesNotExist:
            return Response({'Error' : 'Not Found'}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = WatchListSerializer(movie)
        return Response(serializer.data)

    def put(self, request, pk):
        movie = WatchList.objects.get(pk=pk)
        serializer = WatchListSerializer(instance=movie ,data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        else:
            return Response(serializer.errors)

    def delete(self, request, pk):
        movie = WatchList.objects.get(pk=pk)
        movie.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class StreamPlatformVS(viewsets.ViewSet): #one view for 2 routes 'platform'/'platform/pk'
    permission_classes = [AdminOrReadOnly]

    def list(self, request):
        queryset = StreamPlatform.objects.all()
        serializer = StreamPlatformSerializer(queryset, many=True) #many instances
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = StreamPlatform.objects.all()
        watchlist = get_object_or_404(queryset, pk=pk) #single instance
        serializer = StreamPlatformSerializer(watchlist)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def create(self, request):
        serializer = StreamPlatformSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors)

    def destroy(self, request, pk):
        platform = StreamPlatform.objects.get(pk=pk)
        platform.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
        



class PlatformDetailsAV(APIView):
    permission_classes = [AdminOrReadOnly]

    def get(self, request, pk):
        try:
            platform = StreamPlatform.objects.get(pk=pk)
        except StreamPlatform.DoesNotExist:
            return Response({'Error' : 'Not Found'}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = StreamPlatformSerializer(platform)
        return Response(serializer.data)

    def put(self, request, pk):
        platform = StreamPlatform.objects.get(pk=pk)
        serializer = StreamPlatformSerializer(instance=platform ,data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        else:
            return Response(serializer.errors)

    def delete(self, request, pk):
        platform = StreamPlatform.objects.get(pk=pk)
        platform.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)



class ReviewPerUser(generics.ListAPIView):
    serializer_class = ReviewSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        username = self.request.query_params.get('username')
        return Review.objects.filter(reviewer__username=username)



class ReviewList(generics.ListAPIView):

    serializer_class = ReviewSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [filters.SearchFilter, DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['active']
    ordering_fields = ['rating']
    search_fields = ['rating', 'created', 'reviewer__username']

    def get_queryset(self):
        pk = self.kwargs['pk']
        return Review.objects.filter(watchlist=pk)



class ReviewCreate(generics.CreateAPIView):
    serializer_class = ReviewSerializer
    permission_classes = [IsAuthenticated]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'review-create'
    
    def get_queryset(self):
        return Review.objects.all()
    
    def perform_create(self, serializer):
        pk = self.kwargs.get('pk')
        media = WatchList.objects.get(pk=pk)
        
        reviewer = self.request.user
        review_queryset = Review.objects.filter(reviewer=reviewer, watchlist=media)
        
        if review_queryset.exists():
            raise ValidationError("you have a review for this movie already!")

        # Calculating the average rating
        current_count = media.count_rating
        new_rating = serializer.validated_data['rating']
        if current_count == 0:
            media.avg_rating = new_rating
        else:
            old_total_sum = current_count * media.avg_rating
            media.avg_rating = (old_total_sum + new_rating)/(current_count + 1)

        media.count_rating = current_count + 1
        media.save()

        serializer.save(watchlist=media, reviewer=reviewer)




class ReviewDetails(mixins.RetrieveModelMixin, mixins.UpdateModelMixin, mixins.DestroyModelMixin, generics.GenericAPIView ):
    queryset = Review.objects.all()
    serializer_class = ReviewSerializer
    permission_classes = [ReviewerOrReadOnly]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'review-details'

    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)
    
    def put(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)
    
    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)
    