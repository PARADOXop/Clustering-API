import re
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from get_data.utils import parseBody
import csv
import pandas as pd

from get_data.models import Clusters
# Create your views here.

@csrf_exempt
def clusters(request):
    df_test1 = parseBody(request)
    print("here1")

    return Clusters.getClusters(df_test1)