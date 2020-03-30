from django.conf.urls import url
from .views import (index, show_content, ch01, ch02, ch03)

app_name = "introml"

urlpatterns = [
    url(r'^$', index, name='index'),
    url(r'^show_content/$', show_content, name='show_content'),
    url(r'^ch01/$', ch01, name='ch01'),
    url(r'^ch02/$', ch02, name='ch02'),
    url(r'^ch03/$', ch03, name='ch03'),
]


