# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:05:24 2022

@author: Desenvolvimento3
"""

# API imports

from django.http.response import StreamingHttpResponse, HttpResponse

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

import os
from tkinter import N
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pyModbusTCP.client import ModbusClient
from scipy.signal import savgol_filter
import tensorflow as tf
import warnings
import imagiz
import time

warnings.filterwarnings('ignore')

mpl.rcParams['figure.figsize'] = (16, 8)
mpl.rcParams['axes.grid'] = True
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams["legend.loc"] = 'best'

# client for the video streaming
client = imagiz.Client("cc1", server_ip="localhost")


class client_modbusTCP:
    def __init__(self, ip_address, port):
        self.ip_address = ip_address
        self.port = port

        self.client = ModbusClient(host=self.ip_address, port=self.port)
        self.client.open()

        res = 0

        while (res != 1):
            tmp = self.client.read_holding_registers(9105, 1)
            if (tmp is not None):
                res = tmp[0]
            print("Verificando registradores... {} {}".format(tmp, res))

        print("Comunicação iniciada...\n")

    def write_control_signal(self, shift):
        if (shift < 0):
            shift = 65535 - np.abs(shift)

        res = [self.client.read_holding_registers(9105, 1),
               self.client.write_multiple_registers(
                   9120, [int(shift), 0, 0, 0, 0])]

        print('Escrita: {} - {} - {}'.format(int(shift), res[0], res[1]))

        return res

    def close(self):
        self.client.close()


class PID_control:
    def __init__(self, kp, ki, kd, sat):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.sat = sat

        self.Ts = 1/30
        self.N = 10

        self.control = np.array([])

        self.x = np.array([0])
        self.e = np.array([0])  # Proportional action
        self.E = np.array([0])  # Integral action
        self.de = np.array([0])  # Derivative action
        self.dehat = np.array([0])  # Low pass filtered derivative action

    def update_control(self, x, Ts, enable):
        self.Ts = Ts

        self.x = np.append(self.x, x)
        self.e = np.append(self.e, -x*enable)
        self.de = np.append(self.de, -(self.x[-1] - self.x[-2])*enable)

        dehat = enable*(self.dehat[-1] + self.N*self.de[-1])/(1 + self.N*Ts)
        self.dehat = np.append(self.dehat, dehat)

        if (np.abs(self.e[-1]) <= 10):
            self.E = np.append(
                self.E, enable*(self.E[-1] + self.Ts*self.e[-1]))
        else:
            self.X = np.append(self.E, 0)

        shift = self.kp*self.e[-1] + self.ki * \
            self.E[-1] + self.kd*self.dehat[-1]
        shift = np.clip(shift, -self.sat, self.sat)
        self.control = np.append(self.control, shift)

        return shift

    def log(self):
        print('\nCONTROLE PID')
        print('Frame: {}'.format(self.control.size))
        print('Período de amostragem: T = {}'.format(self.Ts))

        control_law = 'Controlador: C(z) = {} {} {} {}Tz/(z - 1) {} {}H(z)(z - 1)/z'.format(
            '+' if (self.kp >= 0) else '-', np.abs(self.kp),
            '+' if (self.ki >= 0) else '-', np.abs(self.ki),
            '+' if (self.kd >= 0) else '-', np.abs(self.kd))

        print(control_law)

        print('Erro: {}'.format(self.x[-1]))
        print('Ação proporcional: {}'.format(self.kp*self.e[-1]))
        print('Ação derivativa: {}'.format(self.kd*self.dehat[-1]))
        print('Ação integral: {}'.format(self.ki*self.E[-1]))
        print('Controle: u(t) = {}'.format(round(self.control[-1])))

    def plot_debug(self):
        _ = plt.figure()
        plt.plot(self.control, label='Sinal de controle')
        plt.plot(self.kp*self.e, '--', label='Ação proporcional')
        plt.plot(self.ki*self.E, '--', label='Ação integral')
        plt.plot(self.kd*self.dehat, '--', label='Ação derivativa')
        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Deslocamento (mm)')
        plt.title('Controle PID')
        plt.show()


class joint_classifier:
    def __init__(self, filenames, img_width, img_height, class_names):
        self.img_width = img_width
        self.img_height = img_height

        self.models = []

        for f in filenames:
            self.models.append(self.load_model(f))

        self.class_names = class_names

    def load_model(self, filename):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=[self.img_width, self.img_height, 3]),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=11, strides=4, activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=5, activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, activation='relu'),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False),
            metrics=['accuracy']
        )

        model.load_weights(filename)

        return model

    def predict(self, img, comp=1):
        res = tf.image.convert_image_dtype(img, tf.float32)
        res = tf.image.resize(res, [self.img_width, self.img_height])
        batch_image = tf.reshape(res, [1, self.img_width, self.img_height, 3])

        votes = np.zeros(len(self.class_names))

        for model in self.models:
            pred = model.predict(batch_image)[0]
            votes[int(np.argmax(pred))] += 1

        label = self.class_names[np.argmax(votes)]

        return label, (label == self.class_names[comp])


class joint_tracker:
    def __init__(self, source,
                 blur_kernel_size_1=17, blur_kernel_size_2=17,
                 savgol_window_size_1=27, savgol_window_size_2=27,
                 alpha_1=.9, alpha_2=.9, alpha_3=.9, alpha_4=.9,
                 control_params=(.5, .001, .2, 20)):

        self.blur_kernel_size_1 = blur_kernel_size_1
        self.blur_kernel_size_2 = blur_kernel_size_2
        self.savgol_window_size_1 = savgol_window_size_1
        self.savgol_window_size_2 = savgol_window_size_2
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.alpha_4 = alpha_4

        self.cam = cv2.VideoCapture(
            'C:/desenvolvimento-web/testesReactPython/DjangoApi/fsttest/apipwreye/pwreyev309/video_01.avi')
        self.set_ROI()
        self.width, self.height = self.dim[:2]
        self.h, self.w = self.height//2, self.width//2

        self.img_EWMA = np.zeros((self.height, self.width, 3))
        self.p = np.array([self.w, self.h]).reshape(-1, 1)
        self.p_EWMA = np.array([self.w, self.h]).reshape(-1, 1)

        self.bright_line = np.array([])
        self.bright_line_EWMA = np.array([0])

        self.laser_profile_x = []
        self.laser_profile_y = []
        self.laser_profile_x_EWMA = [np.zeros(300)]
        self.laser_profile_y_EWMA = [np.zeros(300)]
        self.dy = []
        self.laser_profile = None

        self.lines = None
        self.a, self.b = 1, 0

        self.enable_control = np.array([])
        self.control = np.array([])

        self.timer = 0

        """
            Variáveis do plotador
        """
        self.plot_x = np.zeros(100)
        self.plot_y = np.zeros(100)
        self.plot_control = np.zeros(100)
        self.control_state = ['Desabilitado', 'Habilitado']
        self.colors = [(0, 0, 255), (0, 255, 0)]

        kp, ki, kd, sat = control_params
        self.controller = PID_control(*control_params)

        if (self.laser2d):
            self.method = self.laser_line
            self.debug = self.laser_profile_debug
        else:
            self.method = self.feature_lines
            self.debug = self.feature_lines_debug

    def read_frame(self):
        self.current_status, self.current_frame = self.cam.read()

        return self.current_frame

    def crop_frame(self):
        self.cropped_frame = self.current_frame[
            self.bbox[1]:(self.bbox[1] + self.bbox[3]),
            self.bbox[0]:(self.bbox[0] + self.bbox[2]), :]
        self.cropped_frame = cv2.resize(self.cropped_frame,
                                        self.dim,
                                        interpolation=cv2.INTER_AREA)

    def set_ROI(self):
        self.current_status, self.current_frame = self.cam.read()
        self.dim = (self.current_frame.shape[1], self.current_frame.shape[0])

        filenames = ['C:/desenvolvimento-web/testesReactPython/DjangoApi/fsttest/apipwreye/pwreyev309/training/casper.ckpt', 'C:/desenvolvimento-web/testesReactPython/DjangoApi/fsttest/apipwreye/pwreyev309/training/balthazar.ckpt',
                     'C:/desenvolvimento-web/testesReactPython/DjangoApi/fsttest/apipwreye/pwreyev309/training/melchior.ckpt']
        class_names = ['topo', 'canto/sobreposta']

        self.jc = joint_classifier(
            filenames, self.dim[1], self.dim[0], class_names)

        _, self.laser2d = self.jc.predict(self.current_frame)

        self.bbox = cv2.selectROI(self.current_frame, False)

        if ((np.array(self.bbox) == 0).all()):
            self.bbox = (0, 0, self.dim[0], self.dim[1])

        self.p1 = (int(self.bbox[0]), int(self.bbox[1]))
        self.p2 = (int(self.bbox[0] + self.bbox[2]),
                   int(self.bbox[1] + self.bbox[3]))

    def convert_ROI_coordinates(self, p):
        x = (self.bbox[2]/self.width) * \
            np.clip(p[0], 0, self.width) + self.bbox[0]
        y = (self.bbox[3]/self.height) * \
            np.clip(p[1], 0, self.height) + self.bbox[1]

        return x, y

    def get_fps(self):
        timer = cv2.getTickCount()
        self.fps = cv2.getTickFrequency()/(timer - self.timer)
        self.timer = timer

        return self.fps

    def get_status(self):
        return self.current_status

    def get_frame(self):
        return self.current_frame

    def get_encoded_frame(self):
        ret, image = self.get_frame
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def gen(camera):
        while True:
            frame = camera.get_encoded_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\\r\n' + frame + b'\r\n\r\n')
            time.sleep(1)

    def release_camera(self):
        self.cam.release()
        cv2.destroyAllWindows()

    def check_camera(self):
        return (cv2.waitKey(1) & 0xFF) != ord('k')

    def get_pos(self):
        return self.p_EWMA[0, -1] - self.w, self.h - self.p_EWMA[1, -1]

    def get_control_status(self):
        return (self.enable_control[-1] == 1)

    def brightest_line(self):
        tmp = self.operated_frame[0].copy()
        row_sum = np.sum(tmp, axis=1)
        y = np.argmax(row_sum)

        self.bright_line = np.append(self.bright_line, y)
        self.bright_line_EWMA = np.append(self.bright_line_EWMA,
                                          self.alpha_1*self.bright_line_EWMA[-1] +
                                          (1 - self.alpha_1)*y)

        self.bright_lines = [self.bright_line_EWMA[-1]]

        return self.bright_line[-1], self.bright_line_EWMA[-1]

    def log_transform(self, img):
        c = 255 / np.log(1 + np.max(img))
        log_img = c*(np.log(img + 1))
        log_img = np.array(log_img, dtype=np.uint8)

        return log_img

    def extract_laser_line(self):
        tmp = cv2.cvtColor(self.img_EWMA, cv2.COLOR_BGR2GRAY)
        tmp = cv2.GaussianBlur(tmp, (self.blur_kernel_size_1,
                                     self.blur_kernel_size_1), 0)

        for i in range(2):
            tmp = tmp[1:-1, :] - tmp[0:-2, :]

        tmp = self.log_transform(tmp)
        tmp = cv2.GaussianBlur(tmp, (self.blur_kernel_size_2,
                                     self.blur_kernel_size_2), 0)
        _, tmp = cv2.threshold(tmp, np.max(tmp)//3,
                               np.max(tmp), cv2.THRESH_BINARY)

        self.operated_frame = [tmp]

        return tmp

    def laser_line(self):
        _ = self.extract_laser_line()

        _, b = self.brightest_line()
        self.b_min = int(np.max([b - 200, 0]))
        self.b_max = int(np.min([b + 200, 479]))

        y, x = np.nonzero(self.operated_frame[0][self.b_min:self.b_max, :])
        res = list(set(np.random.randint(x.size, size=int(x.size/10))))
        x, y = x[res], y[res]

        if (x.size > 2):
            f = interp1d(x, y)
            xhat = np.linspace(np.min(x), np.max(x), num=300)
            yhat = f(xhat)
            yhat = np.nan_to_num(yhat, yhat[1])
            yhat = savgol_filter(yhat,
                                 window_length=self.savgol_window_size_1,
                                 polyorder=2, deriv=0) + self.b_min
        else:
            xhat = np.linspace(0, self.width, 300)
            yhat = np.zeros(300)

        self.laser_profile_x.append(xhat)
        self.laser_profile_y.append(yhat)
        self.laser_profile_x_EWMA.append(self.alpha_2*self.laser_profile_x_EWMA[-1] +
                                         (1 - self.alpha_2)*self.laser_profile_x[-1])
        self.laser_profile_y_EWMA.append(self.alpha_2*self.laser_profile_y_EWMA[-1] +
                                         (1 - self.alpha_2)*self.laser_profile_y[-1])

        xhat, yhat = (self.laser_profile_x_EWMA[-1],
                      self.laser_profile_y_EWMA[-1])

        self.dy.append(savgol_filter(yhat,
                                     window_length=self.savgol_window_size_2,
                                     polyorder=2, deriv=1))

        self.laser_profile = [xhat, yhat, self.dy[-1]]

        res = (np.where(np.diff(np.signbit(self.dy[-1])))[0]).tolist()

        x, y = self.p_EWMA[0, -1], self.p_EWMA[1, -1]

        if (len(res) > 0):
            idx = res[np.argmin(np.abs(xhat[res] - self.w))]
            x, y = self.convert_ROI_coordinates([xhat[idx], yhat[idx]])

        return x, y

    def feature_lines(self):
        gray = cv2.cvtColor(self.img_EWMA, cv2.COLOR_BGR2GRAY)
        tmp = gray[:, 1:-1] - gray[:, 0:-2]
        tmp = cv2.boxFilter(tmp, -1, (13, 13), normalize=True)
        _, tmp = cv2.threshold(tmp, 2*np.max(tmp)//3, np.max(tmp),
                               cv2.THRESH_BINARY)

        col_sum = np.sum(tmp, axis=0)
        x = np.argmax(col_sum)
        max_col_sum = col_sum[x]//np.max(tmp)

        lines = cv2.HoughLines(tmp, 1, np.pi/180,
                               max_col_sum, None, 0, 0)

        _ = self.extract_laser_line()
        _, y = self.brightest_line()

        if (lines is not None):
            lines = lines.ravel()
            r, theta = lines[0::2], lines[1::2]

            a = np.mean(-np.cos(theta)/np.sin(theta))
            b = np.mean(r/np.sin(theta))

            self.a = self.alpha_2*self.a + (1 - self.alpha_2)*a
            self.b = self.alpha_2*self.b + (1 - self.alpha_2)*b

            x = np.int64((y - self.b)/self.a)

        self.operated_frame.append(tmp)
        self.bright_lines = [y, [self.a, self.b]]

        x, y = self.convert_ROI_coordinates([x, y])

        return x, y

    def predict(self):
        self.img_EWMA = np.uint8(self.alpha_4*self.img_EWMA +
                                 (1 - self.alpha_4)*self.cropped_frame)

        x, y = self.method()

        p = np.array([x, y]).reshape(-1, 1)
        p_EWMA = self.alpha_3*self.p_EWMA[:, -1] + (1 - self.alpha_3)*p[:, 0]
        self.p = np.concatenate([self.p, p], axis=1)
        self.p_EWMA = np.concatenate([self.p_EWMA, p_EWMA.reshape(-1, 1)],
                                     axis=1)

        if (self.p_EWMA.shape[1] < 10):
            self.enable_control = np.append(self.enable_control, 0)
        elif (np.var(self.p_EWMA[0, -10:] - self.p[0, -10:]) <= 3):
            self.enable_control = np.append(self.enable_control, 1)
        else:
            self.enable_control = np.append(self.enable_control, 0)

        return self.p_EWMA[0, -1], self.p_EWMA[1, -1]

    def update_control(self):
        x, _ = self.get_pos()
        Ts = 1/self.get_fps()
        enable = self.enable_control[-1]

        shift = np.round(self.controller.update_control(x, Ts, enable))
        self.controller.log()
        self.control = np.append(self.control, shift)

        return shift

    def plot_lines(self, signal, text, color, ylim):
        x = np.linspace(40, self.width - 40, signal.size, dtype=np.int64)

        blank_img = 255*np.ones((self.height, self.width, 3), dtype=np.uint8)

        draw_points = (np.asarray([x, signal]).T).astype(np.int64)
        x_circle, y_circle = draw_points[-1]

        cv2.line(blank_img, (0, self.h), (self.width, self.h), 0, 1)
        cv2.line(blank_img, (100, 0), (100, self.height), 0, 1)
        cv2.polylines(blank_img, [draw_points], False, (255, 0, 0))
        cv2.circle(blank_img, (x_circle, y_circle), 10, color, -1)

        pos = 50
        font = cv2.FONT_HERSHEY_SIMPLEX

        for t in text:
            cv2.putText(blank_img, t, (10, pos), font, 0.5,
                        (255, 0, 0), 2)
            pos += 20

        cv2.putText(blank_img, str(ylim), (100, 140), font, 0.5, (0, 0, 0), 0)
        cv2.putText(blank_img, str(-ylim), (100, 340), font, 0.5, (0, 0, 0), 0)

        return blank_img

    def imshow(self):
        img = self.current_frame.copy()

        idx = int(self.enable_control[-1])
        color = self.colors[idx]
        state = self.control_state[idx]

        x, y = np.round(self.get_pos())

        self.plot_control = np.append(self.plot_control, self.control[-1])[1:]
        self.plot_x = np.append(self.plot_x, x)[1:]
        self.plot_y = np.append(self.plot_y, y)[1:]

        cv2.rectangle(img, self.p1, self.p2, (255, 255, 0), 2, 1)
        cv2.line(img, (0, self.h), (self.width, self.h), (255, 0, 0), 2)
        cv2.line(img, (self.w, 0), (self.w, self.height), (0, 0, 255), 2)

        cv2.circle(img, (int(self.plot_x[-1] + self.w),
                         int(self.h - self.plot_y[-1])), 10, color, -1)
        cv2.circle(img, (self.w, self.h), 3, [0, 255, 255], -1)

        text = [
            "Deslocamento (x, y): {}".format(str(int(self.plot_x[-1])) +
                                             ', ' + str(int(self.plot_y[-1]))),
            "Controle: " + state,
            "FPS: {}".format(str(int(self.fps)))
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX

        pos = 50

        for t in text:
            cv2.putText(img, t, (10, pos), font, 0.5, color, 2)
            pos += 20

        shift = self.plot_control[-1]
        max_ctrl = np.max(np.abs(self.plot_control))
        control_output = np.int64(-100*self.plot_control /
                                  np.max([max_ctrl, 1]) + self.h)

        text = ["Sinal de controle (mm): {}".format(shift)]

        new_img = self.plot_lines(control_output, text, color, max_ctrl)
        img = np.concatenate((img, new_img), axis=1)

        x_max = np.clip(np.max(np.abs(self.plot_x)), 1, self.width)
        x_output = self.h - (100/x_max)*self.plot_x

        text = ['Coordenada x (pixels): {}'.format(round(self.plot_x[-1]))]

        new_img = self.plot_lines(x_output, text, color, round(x_max))
        tmp = new_img.copy()

        y_max = np.clip(np.max(np.abs(self.plot_y)), 1, self.height)
        y_output = self.h - (100/y_max)*self.plot_y

        text = ['Coordenada y (pixels): {}'.format(round(self.plot_y[-1]))]

        new_img = self.plot_lines(y_output, text, color, round(y_max))

        new_img = np.concatenate((tmp, new_img), axis=1)
        img = np.concatenate((img, new_img), axis=0)

        height, width = img.shape[0:2]
        dim = (int(.8*width), int(.8*height))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

        cv2.imshow('camera - results', img)
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)

        return img

    def laser_profile_debug(self):
        op_frame = self.operated_frame[0]
        b_line = self.bright_lines[0]
        img = cv2.cvtColor(self.img_EWMA, cv2.COLOR_BGR2GRAY)

        tmp = img.copy()
        img = img[:op_frame.shape[0], :]
        img = np.concatenate((img, op_frame), axis=1)

        cv2.line(tmp, (0, int(b_line)), (self.width, int(b_line)),
                 (0, 0, 0), 2)
        draw_points = (
            np.asarray([self.laser_profile[0],
                        self.laser_profile[1]]).T).astype(np.int64)
        cv2.polylines(tmp, [draw_points], False, (0, 0, 0))

        blank_img = 255*np.ones((self.height, self.width), dtype=np.uint8)

        if (len(self.laser_profile[2]) > 0):
            draw_points = (
                np.asarray([self.laser_profile[0],
                            self.h + 40*self.laser_profile[2]]).T).astype(
                                np.int64)
            cv2.polylines(blank_img, [draw_points], False, 0)
            cv2.line(blank_img, (0, self.h), (self.width, self.h), 0, 1)

        tmp = np.concatenate((tmp, blank_img), axis=1)
        img = np.concatenate((img, tmp[:img.shape[0], :]), axis=0)

        h, w = img.shape[0:2]
        dim = (int(.8*w), int(.8*h))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

        cv2.imshow('camera - debug', img)

        return img

    def feature_lines_debug(self):
        img = cv2.cvtColor(self.img_EWMA, cv2.COLOR_BGR2GRAY)

        tmp = img.copy()
        img = np.concatenate((img, self.operated_frame[1]), axis=1)

        y, coef = self.bright_lines

        pt1 = (int((-1000 - coef[1])/coef[0]), -1000)
        pt2 = (int((1000 - coef[1])/coef[0]), 1000)

        cv2.line(tmp, pt1, pt2, (0, 0, 0), 2)
        cv2.line(tmp, (0, int(y)), (self.width, int(y)), (0, 0, 0), 2)

        tmp = np.concatenate((self.operated_frame[0],
                              tmp[:self.operated_frame[0].shape[0], :]),
                             axis=1)
        img = np.concatenate((img, tmp[:, :img.shape[1]]), axis=0)

        h, w = img.shape[0:2]
        dim = (int(.8*w), int(.8*h))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

        cv2.imshow('camera - debug', img)

        return img

    def imshow_debug(self):
        img = self.debug()
        img = cv2.resize(np.uint8(img), (640, 480),
                         interpolation=cv2.INTER_CUBIC)
        img = np.stack((img,)*3, axis=-1)

        return img

    def plot_debug(self, control=None):
        self.controller.plot_debug()

        _, ax = plt.subplots(1, 2)
        ax[0].plot(self.p[0, :] - self.w, label='Coordenada x',
                   color='orange')
        ax[0].plot(self.p_EWMA[0, :] - self.w, linestyle=(0, (5, 5)),
                   color='b', label='Coordenada x (EWMA - ' +
                   str(self.alpha_3) + ')')
        ax[0].plot(self.h - self.p[1, :], label='Coordenada y', color='r')
        ax[0].plot(self.h - self.p_EWMA[1, :], linestyle=(0, (5, 5)),
                   color='g', label='Coordenada y (EWMA - ' +
                   str(self.alpha_3) + ')')
        ax[0].legend()
        ax[0].set(xlabel='Frame', ylabel='Deslocamento (pixels)')
        ax[1].plot(self.control, label='Controle')
        ax[1].set(xlabel='Frame',
                  ylabel='Deslocamento (mm)')
        ax[1].legend()
        plt.show()


class file_handler:
    def __init__(self, keys):
        self.keys = keys
        self.directory = os.path.abspath(os.getcwd())
        self.files = self.get_param_files()
        self.d = None

    def get_param_files(self):
        files = []

        for file in os.listdir(self.directory):
            if file.endswith('.txt'):
                files.append(os.path.join('', file))

        return files

    def get_params(self, filename):
        d = dict()

        if (filename == 'k'):
            for param in self.keys:
                value = input(param + ': ')
                d[param] = value if ('str' in param) else eval(value)

            res = input('\nDeseja salvar os parâmetros? [s/n]: ')

            if (res == 's'):
                tmp = [(param + '&' +
                        str(d[param])) for param in self.keys]

                filename = input('Digite o nome do arquivo: ')

                while (filename in self.files):
                    filename = input(
                        'O nome informado já está em uso, por favor forneça outro: ')

                print('\n')

                with open(filename, 'w') as f:
                    f.write('\n'.join(tmp))
                f.close()

            self.d = d

        if (filename in self.files):
            with open(filename, 'r') as f:
                params = [x.split('\n')[0] for x in f.readlines()]
            f.close()

            for p in params:
                param, value = p.split('&')
                d[param] = value if ('str' in param) else eval(value)

            self.d = d
        else:
            print('Erro: arquivo inexistente')

    def change_param(self, param):
        if (param in self.keys):
            val = input('Digite o valor do parâmetro: ')
            val = val if ('str' in param) else eval(val)

            self.d[param] = val

            res = input('\nDeseja salvar os novos parâmetros? [s/_]: ')

            if (res == 's'):
                tmp = [(param + '&' +
                        str(self.d[param])) for param in self.keys]

                filename = input('Digite o nome do arquivo: ')

                while (filename in self.files):
                    filename = input(
                        'O nome informado já está em uso, por favor forneça outro: ')

                with open(filename, 'w') as f:
                    f.write('\n'.join(tmp))
                f.close()

    def run(self):
        print('\nPOWEREYE VERSÃO 3.7\n')

        while (self.d is None):
            print('Arquivos de parâmetros disponíveis:\n')

            for file in self.files:
                print(file)

            filename = input(
                '\nDigite o nome de um arquivo de parâmetros ou k para ajustá-los manualmente: ')

            self.get_params(filename)

        print('\nPARÂMETROS DO SEGUIDOR DE JUNTA:\n')

        while (True):
            for param in self.keys:
                print(param + ': ' + str(self.d[param]))

            param = input('\nDigite o nome do parâmetro ou k para continuar: ')

            if (param == 'k'):
                break

            self.change_param(param)

        if ((self.d['str_source'] == np.array(['0', '1'])).any()):
            self.d['str_source'] = int(self.d['str_source'])

        return self.d


class PowerEyeCamApiView(APIView):

    def get(self, request):

        # These must come from the frontend (POST) method
        keys = ['str_source',
                'blur_kernel_size_1', 'blur_kernel_size_2',
                'savgol_window_size_1', 'savgol_window_size_2',
                'alpha_1', 'alpha_2', 'alpha_3', 'alpha_4',
                'control_params',
                'str_out1', 'str_out2', 'str_out3',
                'str_IP_address', 'modbus_port', 'start_TCP_client',
                'str_saveinput', 'str_saveoutput']

        # File_handler must be adapted or substituted for a function that reads from database
        fh = file_handler(keys)
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
             "str_out1": "outputs/video_01a.avi",
             "str_out2": "outputs/video_01b.avi",
             "str_out3": "outputs/video_01c.avi",
             "str_IP_address": "000.000.000.000",
             "modbus_port": 0,
             "start_TCP_client": False,
             "str_saveinput": "outputs/input_01.txt",
             "str_saveoutput": "outputs/output_01.txt"
             }

        # args for joint_tracker are received directly from database
        args = list(map(d.get, keys[:-9]))
        jt = joint_tracker(*args)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out1 = cv2.VideoWriter(d['str_out1'], fourcc, 20.0, (640, 480))
        out2 = cv2.VideoWriter(d['str_out2'], fourcc, 20.0, (640, 480))
        out3 = cv2.VideoWriter(d['str_out3'], fourcc, 20.0, (640, 480))

        if (d['start_TCP_client']):
            client = client_modbusTCP(d['str_IP_address'], d['modbus_port'])

        while (jt.check_camera()):
            img = jt.read_frame()

            if (jt.get_status()):
                out1.write(img)

                jt.crop_frame()
                _, _ = jt.predict()
                shift = jt.update_control()
                img = jt.imshow()
                debug_img = jt.imshow_debug()

                out2.write(img)
                out3.write(debug_img)

                if (d['start_TCP_client']):
                    _ = client.write_control_signal(shift)
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

        if (d['start_TCP_client']):
            client.close()
