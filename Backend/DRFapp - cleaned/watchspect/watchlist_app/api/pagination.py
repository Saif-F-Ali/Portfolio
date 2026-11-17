from rest_framework.pagination import PageNumberPagination, LimitOffsetPagination, CursorPagination

class CustomPaginationPNP(PageNumberPagination):
    page_query_param = 'p'
    page_size = 5
    # check other useful params PageNumberPagination > configuration
    # page_size_query_param 

class CustomPaginationLOP(LimitOffsetPagination):
    default_limit = 5
    max_limit = 10
    # check other useful params LimitOffsetPagination > configuration

class CustomPaginationCUP(CursorPagination):
    page_size = 5
    order = 'created'
    # check other useful params CursorPagination > configuration