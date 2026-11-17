from rest_framework import serializers
from watchlist_app.models import WatchList, StreamPlatform, Review
from django.utils.text import slugify
from django.shortcuts import get_object_or_404


STREAM_PLATFORM_FIELDS = ['name', 'about', 'website', 'watchlist']
WATCHLIST_FIELDS = [
    'id', 'title', 'storyline', 'platform', 'active', 'count_rating', 'avg_rating', 'created', 'slug', 'warning'
]
REVIEW_FIELDS = [
    'id', 'reviewer', 'rating', 'description', 'watchlist', 'active', 'created', 'update'
]


class ReviewSerializer(serializers.ModelSerializer):
    reviewer = serializers.StringRelatedField(read_only=True) #model User has STR reper. already
    class Meta:
        model = Review
        fields = REVIEW_FIELDS
        #exclude= ['id']


class WatchListSerializer(serializers.ModelSerializer):
    slug = serializers.SerializerMethodField()
    platform = serializers.CharField(source='platform.name', max_length=22)
    warning = serializers.SerializerMethodField(read_only=True)
    
    class Meta:
        model = WatchList
        fields = WATCHLIST_FIELDS
        #exclude = ['id']
    
    # overriding create and update to account for source='platform.name'

    def create(self, validated_data):
        platform_name = validated_data.pop('platform')['name']
        platform = StreamPlatform.objects.get(name=platform_name)
        watchlist = WatchList.objects.create(platform=platform, **validated_data)
        return watchlist

    
    def update(self, instance, validated_data):
        platform_name = validated_data.pop('platform')['name']
        if platform_name:
            platform = get_object_or_404(StreamPlatform, name=platform_name)
            instance.platform = platform

        for attr, value in validated_data.items():
            setattr(instance, attr, value)

        instance.save()
        return instance

    def get_warning(self, obj):
        # We will pass this from the view, if it exists
        return self.context.get('warning_message', None)

    
    def get_slug(self, obj):
        return slugify(obj.title)
        

    def validate_title(self, value): # field-level validator
        if len(value) < 2:
            raise serializers.ValidationError("Title is too short!")
        return value

    def validate(self, data): # object-level validator
        if data['title'] == data['storyline']:
            raise serializers.ValidationError("Media plot cannot be only it's title!")
        return data


class StreamPlatformSerializer(serializers.ModelSerializer):
    watchlist = WatchListSerializer(many=True, read_only=True)

    class Meta:
        model = StreamPlatform
        fields = STREAM_PLATFORM_FIELDS
        
