# -*- coding: utf-8 -*-
from math import *
import numpy as np
from scipy.fftpack import fft, ifft

#周波数をメル尺度でスケール
def helz2mel(helz):
    return 1000 * log(helz / 1000. + 1, 2)

#高域強調フィルタ
def emphasis_filter(data):
    return data[:2] + [data[i] - 0.97 * data[i-1] - 0.2 * data[i-1]
            for i in range(2, len(data))]

#入力波形→スペクトル変換
def wave2spectle(wave, is_emphasis=True):
    ham_window = np.hamming(len(wave))
    if is_emphasis:
        emphasised = emphasis_filter(wave)
        spectle = np.abs(fft(ham_window * emphasised)).tolist()
    else:
        spectle = np.abs(fft(ham_window * wave)).tolist()
    return spectle

