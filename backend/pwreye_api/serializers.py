from rest_framework import serializers
from .models import Camera, Parameters


class ParametersSerializer(serializers.ModelSerializer):
    class Meta:
        model = Parameters
        fields = [
            "str_source",
            "blur_kernel_size_1",
            "blur_kernel_size_2",
            "savgol_window_size_1",
            "savgol_window_size_2",
            "alpha_1",
            "alpha_2",
            "alpha_3",
            "alpha_4",
            "control_param_1",
            "control_param_2",
            "control_param_3",
            "control_param_4",
            "str_out1",
            "str_out2",
            "str_out3",
            "str_IP_address",
            "modbus_port",
            "start_TCP_client",
            "str_saveinput",
            "str_saveoutput"
        ]


class CameraSerializer(serializers.ModelSerializer):
    class Meta:
        model = Camera
        fields = [
            "image",
        ]
