import numpy as np
from scipy.signal import filtfilt, butter

fs = 2800.0


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, Wn=normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def smtdA(txt_data):
    N = len(txt_data)
    txt_data = txt_data / 1.6
    Ax = np.fft.fft(txt_data)
    fs = 2800.0
    Ax_abs = np.abs(Ax)
    theta = np.angle(Ax)
    cutoff = 600
    condA = butter_lowpass_filtfilt(Ax_abs, cutoff, fs)
    condA_real = condA * np.cos(theta)
    condA_imag = condA * np.sin(theta)

    return condA, condA_real, condA_imag
