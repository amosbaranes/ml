from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path(r'', include('ml.apps.main.urls')),
    path(r'core/', include('ml.apps.core.urls')),
    path(r'todo/', include('ml.apps.todo.urls')),
    path(r'introml/', include('ml.apps.introml.urls')),
    path(r'dbillings/', include('ml.apps.dbillings.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
