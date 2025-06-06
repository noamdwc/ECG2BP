import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import pywt
from PyEMD import EMD
from scipy.signal import hilbert


def plot_spectrogram(df, signal_col='ECG_II', time_col='Time', fs=100):
    ecg_signal = df[signal_col].values
    time = df[time_col].values

    f, t, Sxx = spectrogram(ecg_signal, fs)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity [dB]')
    # plt.show()
    plt.savefig('spectrogram.png')

def plot_wavelet_transform(df, signal_col='ECG_II', time_col='Time', fs=100):
    ecg_signal = df[signal_col].values
    time = df[time_col].values

    coefficients, frequencies = pywt.cwt(ecg_signal, np.arange(1, 128), 'cgau1', sampling_period=1 / fs)

    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(coefficients), extent=[time.min(), time.max(), frequencies.min(), frequencies.max()], cmap='PRGn',
               aspect='auto', vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
    plt.yscale('log')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Wavelet Transform')
    plt.colorbar(label='Coefficient Magnitude')
    # plt.show()
    plt.savefig('wavelet_transform.png')

def plot_emd(df, signal_col='ECG_II', time_col='Time'):
    ecg_signal = df[signal_col].values
    time = df[time_col].values

    emd = EMD()
    IMFs = emd(ecg_signal)

    plt.figure(figsize=(10, 8))
    for i, imf in enumerate(IMFs):
        plt.subplot(len(IMFs), 1, i + 1)
        plt.plot(time, imf)
        plt.title(f'IMF {i + 1}')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.show()


def plot_hht(df, signal_col='ECG_II', time_col='Time', fs=100):
    ecg_signal = df[signal_col].values
    time = df[time_col].values

    emd = EMD()
    IMFs = emd(ecg_signal)

    analytic_signal = hilbert(IMFs)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs

    plt.figure(figsize=(10, 8))
    for i in range(len(IMFs)):
        plt.subplot(len(IMFs), 1, i + 1)
        plt.plot(time[:-1], instantaneous_frequency[i])
        plt.title(f'Instantaneous Frequency IMF {i + 1}')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.show()


def plot_hilbert_spectrum(df, signal_col='ECG_II', time_col='Time', fs=100):
    ecg_signal = df[signal_col].values
    time = df[time_col].values

    # Perform EMD to decompose the signal into IMFs
    emd = EMD()
    IMFs = emd(ecg_signal)

    # Apply Hilbert Transform to each IMF
    analytic_signal = hilbert(IMFs)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs

    # Prepare the time axis for the instantaneous frequency plot
    time_inst_freq = time[:-1]

    # Plot the Hilbert Spectrum
    plt.figure(figsize=(10, 8))
    for i in range(len(IMFs)):
        plt.subplot(len(IMFs), 1, i + 1)
        plt.scatter(time_inst_freq, instantaneous_frequency[i], c=amplitude_envelope[i][:-1], cmap='jet', marker='o')
        plt.colorbar(label='Amplitude')
        plt.title(f'Hilbert Spectrum IMF {i + 1}')
        plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.show()


def plot_emd_hilbert_spectrum(df, signal_col='ECG_II', time_col='Time', fs=100):
    ecg_signal = df[signal_col].values
    time = df[time_col].values

    # Perform EMD to decompose the signal into IMFs
    emd = EMD()
    IMFs = emd(ecg_signal)

    # Apply Hilbert Transform to each IMF
    analytic_signal = hilbert(IMFs)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs

    # Prepare the time axis for the instantaneous frequency plot
    time_inst_freq = time[:-1]

    # Plot the Hilbert Spectrum
    plt.figure(figsize=(15, 10))
    for i in range(len(IMFs)):
        plt.subplot(len(IMFs), 1, i + 1)
        plt.scatter(time_inst_freq, instantaneous_frequency[i], c=amplitude_envelope[i][:-1], cmap='jet', marker='o')
        plt.colorbar(label='Amplitude')
        plt.title(f'Hilbert Spectrum IMF {i + 1}')
        plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.show()


def plot_emd_image(df, signal_col='ECG_II', time_col='Time'):
    ecg_signal = df[signal_col].values
    time = df[time_col].values

    # Perform EMD
    emd = EMD()
    IMFs = emd(ecg_signal)

    # Plot the IMFs as a 2D image
    plt.figure(figsize=(15, 10))
    plt.imshow(IMFs, aspect='auto', extent=[time.min(), time.max(), 1, IMFs.shape[0]], cmap='jet')
    plt.colorbar(label='Amplitude')
    plt.title('EMD: Intrinsic Mode Functions')
    plt.ylabel('IMF Index')
    plt.xlabel('Time [sec]')
    # plt.show()
    plt.savefig('emd_image.png')

def plot_hht_image(df, signal_col='ECG_II', time_col='Time', fs=100):
    instantaneous_frequency, time_inst_freq = hht_trans(df, signal_col, time_col, fs)

    # Plot the instantaneous frequency of IMFs as a 2D image
    plt.figure(figsize=(15, 10))
    plt.imshow(instantaneous_frequency, aspect='auto',
               extent=[time_inst_freq.min(), time_inst_freq.max(), 1, instantaneous_frequency.shape[0]], cmap='jet')
    plt.colorbar(label='Instantaneous Frequency [Hz]')
    plt.title('HHT: Instantaneous Frequency of IMFs')
    plt.ylabel('IMF Index')
    plt.xlabel('Time [sec]')
    plt.savefig('hht_image.png')

    # plt.show()


def hht_plot_ax(ax, instantaneous_frequency, time_inst_freq, axes_labels=False):
    # Plot the instantaneous frequency of IMFs as a 2D image
    ax.imshow(instantaneous_frequency, aspect='auto',
              extent=(time_inst_freq.min(), time_inst_freq.max(), 1, instantaneous_frequency.shape[0]), cmap='jet')
    if axes_labels:
        ax.title('HHT')
        ax.colorbar(label='Freq[Hz]')
        ax.ylabel('IMF Index')
        ax.xlabel('Time [sec]')



def hht_transform(signal, signal_col='ECG_II', time_col='Time', fs=100):
    ecg_signal = signal[signal_col].values
    time = signal[time_col].values

    # Perform EMD
    emd = EMD()
    IMFs = emd(ecg_signal)

    # Apply Hilbert Transform to each IMF
    analytic_signal = hilbert(IMFs)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
    # Prepare the time axis for the instantaneous frequency plot
    time_inst_freq = time[:-1]
    return instantaneous_frequency, time_inst_freq


def emd_transform(signal, signal_col='ECG_II', time_col='Time'):
    ecg_signal = signal[signal_col].values
    time = signal[time_col].values

    # Perform EMD
    emd = EMD()
    IMFs = emd(ecg_signal)

    return IMFs, time


def emd_plot_ax(ax, IMFs, time, axes_labels=False):
    # Plot the IMFs as a 2D image
    ax.imshow(IMFs, aspect='auto', extent=(time.min(), time.max(), 1, IMFs.shape[0]), cmap='jet')
    if axes_labels:
        ax.title('EMD')
        ax.ylabel('IMF Index')
        ax.xlabel('Time [sec]')
        ax.colorbar(label='Amplitude')


def spec_transform(signal, signal_col='ECG_II', time_col='Time', fs=100):
    ecg_signal = signal[signal_col].values
    time = signal[time_col].values

    f, t, Sxx = spectrogram(ecg_signal, fs)

    return f, t, Sxx


def spectrogram_plot_ax(ax, f, t, Sxx, axes_labels=False):
    ax.pcolormesh(t, f, 10 * np.log10(Sxx))
    if axes_labels:
        ax.title('Spec')
        ax.ylabel('Frequency [Hz]')
        ax.xlabel('Time [sec]')
        ax.colorbar(label='Intensity [dB]')



def wavelet_transform(signal, signal_col='ECG_II', time_col='Time', fs=100):
    ecg_signal = signal[signal_col].values
    time = signal[time_col].values

    coefficients, frequencies = pywt.cwt(ecg_signal, np.arange(1, 128), 'cgau1', sampling_period=1 / fs)

    return coefficients, frequencies, time


def wavelet_transform_plot_ax(ax, coefficients, frequencies, time, axes_labels=False):
    ax.imshow(np.abs(coefficients), extent=(time.min(), time.max(), frequencies.min(), frequencies.max()), cmap='PRGn',
              aspect='auto', vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
    if axes_labels:
        ax.title('Wav')
        ax.ylabel('Frequency [Hz]')
        ax.xlabel('Time [sec]')
        ax.colorbar(label='Coefficient Magnitude')
