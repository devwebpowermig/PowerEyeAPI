from django.http.response import StreamingHttpResponse, HttpResponse

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import Parameters
from .serializers import ParametersSerializer
from pwreyev309.PowerEye_v03 import joint_tracker

import cv2
import threading
import numpy as np
# Create your views here.


class ParametersListApiView(APIView):

    def get(self, request, *args, **kwargs):
        '''List all the parameters'''

        parameters = Parameters.objects.all()
        serializer = ParametersSerializer(parameters, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        '''Creates a new custom parameters list'''

        data = {
            "str_source": request.data.get("str_source"),
            "blur_kernel_size_1": request.data.get("blur_kernel_size_1"),
            "blur_kernel_size_2": request.data.get("blur_kernel_size_2"),
            "savgol_window_size_1": request.data.get("savgol_window_size_1"),
            "savgol_window_size_2": request.data.get("savgol_window_size_2"),
            "alpha_1": request.data.get("alpha_1"),
            "alpha_2": request.data.get("alpha_2"),
            "alpha_3": request.data.get("alpha_3"),
            "alpha_4": request.data.get("alpha_4"),
            "control_param_1": request.data.get("control_param_1"),
            "control_param_2": request.data.get("control_param_2"),
            "control_param_3": request.data.get("control_param_3"),
            "control_param_4": request.data.get("control_param_4"),
            "str_out1": request.data.get("str_out1"),
            "str_out2": request.data.get("str_out2"),
            "str_out3": request.data.get("str_out3"),
            "str_IP_address": request.data.get("str_IP_address"),
            "modbus_port": request.data.get("modbus_port"),
            "start_TCP_client": request.data.get("start_TCP_client"),
            "str_saveinput": request.data.get("str_saveinput"),
            "str_saveoutput": request.data.get("str_saveoutput"),
        }
        serializer = ParametersSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(data, status=status.HTTP_400_BAD_REQUEST)


class ParametersDetailApiView(APIView):

    def get_object(self, parameters_id):
        '''Helper method to get a list of parameters based on an id'''

        try:
            return Parameters.objects.get(id=parameters_id)
        except Parameters.DoesNotExist:
            return None
    # retrieves data from a specific paramter list

    def get(self, request, parameters_id, *args, **kwargs):
        '''Retrieves the list with given id'''
        parameters_instance = self.get_object(parameters_id)
        if not parameters_instance:
            return Response(
                {"res": "Object with this id does not exist"},
                status=status.HTTP_400_BAD_REQUEST
            )

        serializer = ParametersSerializer(parameters_instance)
        return Response(serializer.data, status=status.HTTP_200_OK)
    # updates paramters of a particular list

    def put(self, request, parameters_id, *args, **kwargs):
        parameters_instance = self.get_object(parameters_id)
        if not parameters_instance:
            return Response(
                {"res": "Object with this id does not exist"},
                status=status.HTTP_400_BAD_REQUEST
            )
        data = {
            "str_source": request.data.get("str_source"),
            "blur_kernel_size_1": request.data.get("blur_kernel_size_1"),
            "blur_kernel_size_2": request.data.get("blur_kernel_size_2"),
            "savgol_window_size_1": request.data.get("savgol_window_size_1"),
            "savgol_window_size_2": request.data.get("savgol_window_size_2"),
            "alpha_1": request.data.get("alpha_1"),
            "alpha_2": request.data.get("alpha_2"),
            "alpha_3": request.data.get("alpha_3"),
            "alpha_4": request.data.get("alpha_4"),
            "control_param_1": request.data.get("control_param_1"),
            "control_param_2": request.data.get("control_param_2"),
            "control_param_3": request.data.get("control_param_3"),
            "control_param_4": request.data.get("control_param_4"),
            "str_out1": request.data.get("str_out1"),
            "str_out2": request.data.get("str_out2"),
            "str_out3": request.data.get("str_out3"),
            "str_IP_address": request.data.get("str_IP_address"),
            "modbus_port": request.data.get("modbus_port"),
            "start_TCP_client": request.data.get("start_TCP_client"),
            "str_saveinput": request.data.get("str_saveinput"),
            "str_saveoutput": request.data.get("str_saveoutput"),
        }
        serializer = ParametersSerializer(
            instance=parameters_instance, data=data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # Delete
    def delete(self, request, parameters_id, *args, **kwargs):
        '''Deletes the parameters list if it exists'''
        parameters_instance = self.get_object(parameters_id)
        if not parameters_instance:
            return Response(
                {"res": "Object with this id does not exist"},
                status=status.HTTP_400_BAD_REQUEST
            )
        parameters_instance.delete()
        return Response(
            {"res": "Object deleted successfully!"},
            status=status.HTTP_200_OK
        )


class VideoCamApiView(APIView):

    '''Runs the PowerEye_v03 application'''
    try:

        def get(self, request):

            d = {"str_source": "video_01.avi",
                 "blur_kernel_size_1": 27,
                 "blur_kernel_size_2": 27,
                 "savgol_window_size_1": 17,
                 "savgol_window_size_2": 17,
                 "alpha_1": 0.9,
                 "alpha_2": 0.9,
                 "alpha_3": 0.9,
                 "alpha_4": 0.9,
                 "control_params": (0.5, 0.01, 0.2),
                 "str_out1": "C:/desenvolvimento-web/testesReactPython/DjangoApi/fsttest/apipwreye/pwreyev309/outputs/video_01a.avi",
                 "str_out2": "C:/desenvolvimento-web/testesReactPython/DjangoApi/fsttest/apipwreye/pwreyev309/outputs/video_01b.avi",
                 "str_out3": "C:/desenvolvimento-web/testesReactPython/DjangoApi/fsttest/apipwreye/pwreyev309/outputs/video_01c.avi",
                 "str_IP_address": "000.000.000.000",
                 "modbus_port": 0,
                 "start_TCP_client": False,
                 "str_saveinput": "C:/desenvolvimento-web/testesReactPython/DjangoApi/fsttest/apipwreye/pwreyev309/outputs/input_01.txt",
                 "str_saveoutput": "C:/desenvolvimento-web/testesReactPython/DjangoApi/fsttest/apipwreye/pwreyev309/outputs/output_01.txt"
                 }

            jt = joint_tracker(
                source="C:/desenvolvimento-web/testesReactPython/DjangoApi/fsttest/apipwreye/pwreyev309/video_01.avi")

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out1 = cv2.VideoWriter(d['str_out1'], fourcc, 20.0, (640, 480))
            out2 = cv2.VideoWriter(d['str_out2'], fourcc, 20.0, (640, 480))
            out3 = cv2.VideoWriter(d['str_out3'], fourcc, 20.0, (640, 480))

            while (jt.check_camera()):
                img = jt.read_frame()

                if (jt.get_status()):
                    out1.write(img)
                    jt.crop_frame()
                    _, _ = jt.predict()
                    shift = jt.update_control()
                    img = jt.imshow()

                    # StreamingHttpResponse(
                    #     (encoded_img), content_type='multipart/x-mixed-replace; boundary=frame')

                    debug_img = jt.imshow_debug()

                    out2.write(img)
                    out3.write(debug_img)

                else:
                    break

                np.savetxt(d['str_saveinput'],  jt.control, delimiter='\n')
                np.savetxt(d['str_saveoutput'],
                           jt.p_EWMA[0, :] - jt.w, delimiter='\n')

                out1.release()

                out2.release()
                out3.release()
                jt.plot_debug()
                jt.release_camera()

    except:
        print("Could not open PowerEye_v03 application")


class ImageStreaming(object):

    def __init__(self):
        self.video = cv.VideoCapture(1)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
