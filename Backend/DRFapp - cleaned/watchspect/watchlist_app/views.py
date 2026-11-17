# from django.shortcuts import render
# from watchlist_app.models import Movie
# from django.http import JsonResponse

# # Create your views here. (Endpoints)
# def movie_list(request):
#     movie = Movie.objects.all()
#     data = {
#         'movie' : list(movie.values())
#     }
#     return JsonResponse(data)

# # no decorators like in Flask! the path is added to urls.py file (app-level)
# def movie_details(request, pk):
#     movie = Movie.objects.get(pk=pk)
#     data = {
#         'name' : movie.name,
#         'description' : movie.description,
#         'active' : movie.active
#     }
#     return JsonResponse(data)