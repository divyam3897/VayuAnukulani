from django.shortcuts import render
import requests, json, time
from django.http import JsonResponse
from firebase import firebase
import urllib
from bs4 import BeautifulSoup
from .models import pollutants, classPollutant, regrPollutant
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
import socket
import numpy as np
from django.db.models import Q

def home_page(request):
  return render(request, 'home/home.html')


def get_maps(request):
  entries = pollutants.objects.all()
  return render(request, 'home/map.html', context={'pollutants': entries})

def get_details(request,string):
  entries = pollutants.objects.filter(Q(location=string))
  classes = classPollutant.objects.filter(Q(location=string))
  predictions = regrPollutant.objects.filter(Q(location=string))
  return render(request,
                  "home/details.html",
                  context={'place':string, 'pollutants': entries, "classes": classes, "predictions": predictions})
