# -*- coding: utf-8 -*-
from tools import *
from PySide.QtCore import *
from PySide.QtGui import *
from PySide.QtOpenGL import *
from OpenGL.GL import *
from OpenGL.GLU import *


#グラフ描画クラス
class Plotter(QGLWidget):
    def __init__(self,
            plot_range=(-1, 1, -1, 1), 
            ismel=False,
            ispolygon=False,
            parent=None):
        super(Plotter, self).__init__(parent)
        
        self._data = []
        self.setMinimumSize(1100,300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_range = plot_range
        self.ismel = ismel
        self.ispolygon = ispolygon
        self.grid_width = 1
        self.plot_width = 2
        self.plot_color_table = [
                (0.24, 0.62, 1),
                (0.5, 0.31, 0.12),
                (0.62, 1, 0.24),
                ]

    def initializeGL(self):
        self.qglClearColor(Qt.black)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        depth_range = (-1, 1)
        ortho_range = self.plot_range + depth_range
        glOrtho(*ortho_range)
        self._draw_grid()
        self._draw_data()
        glFlush()
       
    def _draw_data(self):
        for i, d in enumerate(self._data):
            color_idx = min(len(self.plot_color_table)-1, i)
            color = self.plot_color_table[color_idx]
            glColor(*color)
            if self.ispolygon:
                self._draw_data_polygon(d)
            else:
                self._draw_data_line(d)

    #データを線で描画
    def _draw_data_line(self, data):
        glLineWidth(self.plot_width)
        glBegin(GL_LINE_STRIP)
        for x, y in data:
            if self.ismel:
                x = helz2mel(x)
            glVertex2f(x, y)
        glEnd()

    #データをポリゴンで描画
    def _draw_data_polygon(self, data):
        glBegin(GL_TRIANGLE_STRIP)
        xmin, xmax, ymin, ymax = self.plot_range
        glVertex2f(xmin, ymin)
        for i, (x, y) in enumerate(data[:-1]):
            x2, _ = data[i+1]
            if self.ismel:
                x = helz2mel(x)
                x2 = helz2mel(x2)
            glVertex2f(x, y)
            glVertex2f(x2, ymin)
        glVertex2f(*data[-1])
        glEnd()

    #グリッドの描画
    def _draw_grid(self):
        glLineWidth(self.grid_width)
        glColor(0.5, 0.5, 0.5)
        xmin, xmax, ymin, ymax= self.plot_range
        xstep = 10 ** max(1, int((log10(xmax)-1)))
        ystep = 10 ** max(1, int((log10(ymax))))
        glBegin(GL_LINES)
        for x in range(int(xmin), int(xmax), xstep):
            glVertex2f(x, ymin)
            glVertex2f(x, ymax)
        for y in range(int(ymin), int(ymax), ystep):
            glVertex2f(xmin, y)
            glVertex2f(xmax, y)
        glEnd()
        glColor(1, 1, 1)

    #描画データの追加
    def add_data(self, data):
        if isinstance(data[0], (tuple, list)):
            self._data.append(data)
        else:  # x軸を自動付加
            xs = range(0, len(data))
            self._data.append(zip(xs, data))

    #全ての描画データを削除
    def remove_all(self):
        self._data[:] = []
