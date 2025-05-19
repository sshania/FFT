import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import find_peaks, medfilt
from scipy.io import wavfile  # <-- buat baca file .wav

def detect_frequency_from_wav(filepath):
    # Baca file wav
    sample_rate, data = wavfile.read(filepath)
    print(f"Loaded '{filepath}' dengan sample rate {sample_rate} Hz")
    
    # Jika stereo, ambil 1 channel saja
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Normalisasi data
    data = data / np.max(np.abs(data))
    
    # Median filter untuk noise
    data = medfilt(data, kernel_size=3)
    
    # Hann window
    window = np.hanning(len(data))
    data = data * window
    
    # FFT
    N = len(data)
    fft_result = fft(data)
    freqs = np.fft.fftfreq(N, 1/sample_rate)
    
    # Hanya frekuensi positif
    magnitude = np.abs(fft_result[:N//2])
    freqs = freqs[:N//2]
    
    # Filter frekuensi di antara 80 - 350 Hz
    valid_range = (freqs >= 80) & (freqs <= 350)
    freqs = freqs[valid_range]
    magnitude = magnitude[valid_range]
    
    # Cari puncak tertinggi
    peaks, properties = find_peaks(magnitude, height=np.max(magnitude) * 0.3)
    
    if len(peaks) > 0:
        peak_index = peaks[np.argmax(properties['peak_heights'])]
        dominant_freq = freqs[peak_index]
    else:
        dominant_freq = 0.0
    
    # Parabolic interpolation
    if 0 < peak_index < len(freqs) - 1:
        alpha = magnitude[peak_index - 1]
        beta = magnitude[peak_index]
        gamma = magnitude[peak_index + 1]
        peak_adjustment = (alpha - gamma) / (2 * (alpha - 2 * beta + gamma))
        dominant_freq += peak_adjustment * (freqs[1] - freqs[0])
    
    # Visualisasi
    plt.plot(freqs, magnitude)
    plt.xlabel("Frekuensi (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Spektrum Frekuensi")
    plt.grid()
    plt.show()
    
    print(f"Frekuensi Dominan: {dominant_freq:.2f} Hz")
    return dominant_freq


# detect_frequency_from_wav("RekamanGuitar/Senar_Gitar_1.wav")
# detect_frequency_from_wav("RekamanGuitar/Senar_Gitar_2.wav")
# detect_frequency_from_wav("RekamanGuitar/Senar_Gitar_3.wav")
# detect_frequency_from_wav("RekamanGuitar/Senar_Gitar_4.wav")
# detect_frequency_from_wav("RekamanGuitar/Senar_Gitar_5.wav")
# detect_frequency_from_wav("RekamanGuitar/Senar_Gitar_6.wav")


# Coba electric guitar
# detect_frequency_from_wav("RekamanElectricGuitar/ES1.wav")
# detect_frequency_from_wav("RekamanElectricGuitar/ES2.wav")
# detect_frequency_from_wav("RekamanElectricGuitar/ES3.wav")
# detect_frequency_from_wav("RekamanElectricGuitar/ES4.wav")
# detect_frequency_from_wav("RekamanElectricGuitar/ES5.wav")
# detect_frequency_from_wav("RekamanElectricGuitar/ES6.wav")

detect_frequency_from_wav("RekamanViolin/G.wav")
detect_frequency_from_wav("RekamanViolin/D.wav")
detect_frequency_from_wav("RekamanViolin/A.wav")
detect_frequency_from_wav("RekamanViolin/E.wav")