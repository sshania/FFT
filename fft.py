import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

def record_audio(duration=3, fs=44100):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()
    print("Recording complete.")
    return audio.flatten(), fs






def fft_pitch_detection(audio, fs):
    """ Menggunakan FFT untuk mendeteksi fundamental frequency """
    window = np.hanning(len(audio))
    audio = audio * window

    n = 2 ** np.ceil(np.log2(len(audio))).astype(int)

    fft_result = np.fft.fft(audio, n=n)
    freqs = np.fft.fftfreq(n, d=1/fs)

    positive_freqs = freqs[:n//2]
    magnitudes = np.abs(fft_result[:n//2])

    min_freq = 20
    valid_indices = np.where(positive_freqs >= min_freq)
    
    filtered_freqs = positive_freqs[valid_indices]
    filtered_magnitudes = magnitudes[valid_indices]

    peak_index = np.argmax(filtered_magnitudes)
    fundamental_frequency = filtered_freqs[peak_index]

    return fundamental_frequency




duration = 3 
audio, fs = record_audio(duration=duration)


fundamental_freq_fft = fft_pitch_detection(audio, fs)


print(f"Fundamental Frequency (FFT): {fundamental_freq_fft:.2f} Hz")
