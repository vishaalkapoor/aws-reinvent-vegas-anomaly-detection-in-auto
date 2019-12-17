from __future__ import print_function
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.core.pylabtools import figsize
import numpy as np
import pandas as pd
from scipy import signal
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
from ipywidgets import TwoByTwoLayout, Layout
from ipywidgets import Button, Layout
import matplotlib
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 50)

# Matplotlib Settings
plt.style.use('dark_background')
font = {
        'weight' : 'bold',
        'size'   : 16
}
matplotlib.rc('font', **font)

# Plotting Meta data
ECU_META = {
    'engine_load': ('Engine Load/Throttle', 'xkcd:chartreuse'),
    'throttle': ('Engine Load/Throttle', 'xkcd:tomato'),
    'rpm': ('Engine RPM', 'orangered'),
    'speed': ('Speed', 'springgreen'),
    'maf': ('Mass Air Flow Sensor', 'xkcd:azure'),
    'timing_c1': ('Cylinder 1 Timing', 'xkcd:lime'),
    'o2_sensor_voltage': ('O2 sensor voltages', 'xkcd:orange'),
    'o2_sensor_voltage2': ('O2 sensor voltages', 'xkcd:silver')
}

def plot_sensor_data(ax, data, col, **opts):
    """
    Helper method to plot ECU Data
    """
    title, color = ECU_META[col]
    ax.set_title(title, color='#dddddd')
    ax.set_facecolor('#101010')
    x = list(range(data[col].shape[0]))
    ax.plot(x, data[col], color, **opts)


def plot_spectrogram(ax, data, col):
    values = data[col].values
    f, t, Sxx = signal.spectrogram(values, 1.1, signal.windows.gaussian(50,25, sym=True))
    ax.pcolormesh(t, f, Sxx)
    title, color = ECU_META[col]
    ax.set_title(title)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')

def plot_butterworth():
    b, a = signal.butter(5, .01, 'lowpass', analog=False)
    w, h = signal.freqs(b, a)
    plt.plot(np.log(w), (abs(h)))
    #plt.xscale('log')
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [log Hz]]')
    plt.ylabel('Amplitude')
    plt.grid(which='both', axis='both')
    plt.axvline(-10, color='green') # cutoff frequency
    plt.show()


def create_widgets(epochs):
    # Define some widgets for progress indication.
    progress_widget = widgets.IntProgress(
        value=0,
        min=0,
        max=epochs,
        step=1,
        description='Progress:',
        bar_style='success',
        orientation='horizontal',
        layout=Layout(width='1000px', height='45px'))
    mse_widget = widgets.FloatProgress(
        value=0,
        min=0,
        max=0,
        description='Error:',
        bar_style='danger',
        layout=Layout(width='1000px', height='45px'))

    return progress_widget, mse_widget

def setup_mse_plot():
    plt.ion()  # Interactive mode on
    fig = plt.figure()
    fig.set_facecolor('#050505')
    fig.set_figwidth(20)
    fig.set_figheight(10)
    ax = plt.gca()
    
    ax.set_title("Test Error")
    ax.set_ylabel('MSE')
    ax.set_xlabel('Epoch')
    ax.set_facecolor("#101010")

    ax.plot(range(len([])), [])
    fig.show()
    fig.canvas.draw()
    return fig, ax


def plot_mse(fig, ax, mse):
    ax.clear()
    ax.set_title("Test Error")
    ax.set_ylabel('MSE')
    ax.set_xlabel('Epoch')
    fig.set_facecolor('#050505')

    ax.plot(range(len(mse)), mse)
    fig.canvas.draw()


def visualize_anomalies(raw, errors, threshold_constant=2):
    error_threshold =  np.mean(errors) + threshold_constant*np.std(errors)
    anomaly_errors = list(map(lambda v: v > error_threshold, errors))
    # Result visualization
    anomaly = list(map(lambda v: "red" if v else "#101010", anomaly_errors))
    anomaly_bars = list(map(lambda v: 70 if v else 0, anomaly_errors))

    fig, ax = plt.subplots(2, figsize=(30, 15))

    fig.set_facecolor('#050505')

    x = list(range(len(raw["maf"])))

    ax[0].set_title('Engine Readings', color='#dddddd')
    ax[0].set_facecolor('#101010')
    ax[0].plot(x, raw["maf"], 'xkcd:azure')
    #ax[0].plot(x, raw['rpm'], 'orangered')
    ax[0].plot(x, raw["engine_load"],  'xkcd:chartreuse')
    ax[0].plot(x, raw["speed"], 'springgreen')

    #ax[0].plot(x, raw["intake_air_temp"])
    ax[0].plot(x, raw["timing_c1"], 'xkcd:lime')
    ax[0].plot(x, raw["throttle"] ,'xkcd:tomato')

    ax[1].set_title('Anomaly Rating', color='#dddddd')
    ax[1].set_facecolor('#101010')
    ax[1].plot(x, errors/10)
    ax[1].scatter(x, errors/10, c = anomaly)


    plt.show()
