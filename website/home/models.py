from django.db import models
from django.utils import timezone


class pollutants(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    pm25 = models.FloatField(null=True, blank=True)
    pm10 = models.FloatField(null=True, blank=True)
    no2 = models.FloatField(null=True, blank=True)
    so2 = models.FloatField(null=True, blank=True)
    co = models.FloatField(null=True, blank=True)
    temp = models.FloatField(null=True, blank=True)
    hum = models.FloatField(null=True, blank=True)
    location = models.CharField(max_length=100)

    def __str__(self):
        return str(self.date)

class concPollutant(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    pm25 = models.FloatField(null=True, blank=True)
    pm10 = models.FloatField(null=True, blank=True)
    no2 = models.FloatField(null=True, blank=True)
    so2 = models.FloatField(null=True, blank=True)
    co = models.FloatField(null=True, blank=True)
    temp = models.FloatField(null=True, blank=True)
    o3 = models.FloatField(null=True, blank=True)
    hour = models.IntegerField(null=True, blank=True)
    month = models.IntegerField(null=True, blank=True)
    location = models.CharField(max_length=100)

    def __str__(self):
        return str(self.date)

class classPollutant(models.Model):
    date = models.DateTimeField()
    pm25 = models.IntegerField(null=True, blank=True)
    so2 = models.IntegerField(null=True, blank=True)
    no2 = models.IntegerField(null=True, blank=True)
    pm10 = models.IntegerField(null=True, blank=True)
    location = models.CharField(max_length=100)
    def __str__(self):
        return str(self.date)

class regrPollutant(models.Model):
    date = models.DateTimeField()
    pm25 = models.IntegerField(null=True, blank=True)
    so2 = models.IntegerField(null=True, blank=True)
    no2 = models.IntegerField(null=True, blank=True)
    pm10 = models.IntegerField(null=True, blank=True)
    location = models.CharField(max_length=100)

    def __str__(self):
        return str(self.date)
