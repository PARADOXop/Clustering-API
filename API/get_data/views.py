from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from get_data.utils import parseBody

from get_data.models import Clusters
# Create your views here.

@csrf_exempt
def clusters(request):
    body_json = parseBody(request)
    
    return Clusters.getClusters(body_json)