from django.shortcuts import render, redirect, get_object_or_404
from redis import Redis
from .models import (Item)

try:
    redis = Redis(host='redis', port=6379)
except Exception:
    pass


def index(request):
    if request.method == 'POST':
        Item.objects.create(text=request.POST['item_text'])

        # return redirect('/')
    items = Item.objects.all()

    counter = 100

    return render(request, 'todo/index.html', {'items': items, 'x': counter})
