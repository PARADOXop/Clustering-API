import re
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from get_data.utils import parseBody
import json
import pandas as pd

from get_data.models import Clusters
# Create your views here.

@csrf_exempt
def clusters(request):
    df_test0 = parseBody(request)
    print("here1")
    print(type(df_test0))

    return Clusters.getClusters(df_test0)