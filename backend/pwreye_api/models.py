from django.db import models


# Create your models here.

class Camera(models.Model):
    # camera data
    image = models.CharField(max_length=255, blank=False, null=False)


class Parameters(models.Model):
    str_source = models.CharField(max_length=255, blank=True)
    blur_kernel_size_1 = models.FloatField(
        null=True, blank=True, max_length=255)
    blur_kernel_size_2 = models.FloatField(
        null=True, blank=True, max_length=255)
    savgol_window_size_1 = models.FloatField(
        null=True, blank=True, max_length=255)
    savgol_window_size_2 = models.FloatField(
        null=True, blank=True, max_length=255)
    alpha_1 = models.FloatField(null=True, blank=True, max_length=255)
    alpha_2 = models.FloatField(null=True, blank=True, max_length=255)
    alpha_3 = models.FloatField(null=True, blank=True, max_length=255)
    alpha_4 = models.FloatField(null=True, blank=True, max_length=255)
    control_param_1 = models.FloatField(null=True, blank=True, max_length=255)
    control_param_2 = models.FloatField(null=True, blank=True, max_length=255)
    control_param_3 = models.FloatField(null=True, blank=True, max_length=255)
    control_param_4 = models.FloatField(null=True, blank=True, max_length=255)
    str_out1 = models.FloatField(null=True, blank=True, max_length=255)
    str_out2 = models.FloatField(null=True, blank=True, max_length=255)
    str_out3 = models.FloatField(null=True, blank=True, max_length=255)
    str_IP_address = models.CharField(null=True, blank=True, max_length=255)
    modbus_port = models.CharField(null=True, blank=True, max_length=255)
    start_TCP_client = models.CharField(null=True, blank=True, max_length=255)
    str_saveinput = models.CharField(null=True, blank=True, max_length=255)
    str_saveoutput = models.CharField(null=True, blank=True, max_length=255)
