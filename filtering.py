import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import find_peaks, medfilt

def record_audio(duration=3, sample_rate=44100):
    print("Merekam...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("Rekaman selesai.")
    return audio_data.flatten(), sample_rate

def detect_frequency(duration=3, sample_rate=44100):
    data, sample_rate = record_audio(duration, sample_rate)
    data = data / np.max(np.abs(data))

    data = medfilt(data, kernel_size=3)
    

    window = np.hanning(len(data))
    data = data * window

    N = len(data)
    fft_result = fft(data)
    freqs = np.fft.fftfreq(N, 1/sample_rate)
    
    magnitude = np.abs(fft_result[:N//2])
    freqs = freqs[:N//2]
    

    valid_range = (freqs >= 80) & (freqs <= 350)
    freqs = freqs[valid_range]
    magnitude = magnitude[valid_range]
    

    peaks, properties = find_peaks(magnitude, height=np.max(magnitude) * 0.3)  
    
    if len(peaks) > 0:
        peak_index = peaks[np.argmax(properties['peak_heights'])]
        dominant_freq = freqs[peak_index]
    else:
        dominant_freq = 0.0
    

    if 0 < peak_index < len(freqs) - 1:
        alpha = magnitude[peak_index - 1]
        beta = magnitude[peak_index]
        gamma = magnitude[peak_index + 1]
        peak_adjustment = (alpha - gamma) / (2 * (alpha - 2 * beta + gamma))
        dominant_freq += peak_adjustment * (freqs[1] - freqs[0])
    

    plt.plot(freqs, magnitude)
    plt.xlabel("Frekuensi (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Spektrum Frekuensi")
    plt.grid()
    plt.show()
    
    print(f"Frekuensi Dominan: {dominant_freq:.2f} Hz")
    return dominant_freq


detect_frequency()