import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

# Konfigurasi rekaman
DURATION = 3  # Lama rekaman dalam detik
SAMPLING_RATE = 44100  # Frekuensi sampling (Hz)

def record_audio(duration, sampling_rate):
    print("Mulai merekam...")
    audio = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='float32')
    sd.wait()
    print("Rekaman selesai!")
    return audio.flatten()

def compute_fft(audio_signal, sampling_rate):
    N = len(audio_signal)
    fft_values = fft(audio_signal)
    fft_magnitudes = np.abs(fft_values[:N // 2]) 
    frequencies = fftfreq(N, 1 / sampling_rate)[:N // 2]
    
    fft_magnitudes_db = 20 * np.log10(fft_magnitudes + 1e-10) 
    # fft_magnitudes_db = 20 * np.log10(fft_magnitudes + np.finfo(float).eps)
 

    valid_range = frequencies > 20
    # valid_range = (frequencies > 20) & (frequencies < 5000)

    dominant_freq = frequencies[valid_range][np.argmax(fft_magnitudes[valid_range])]
    # dominant_freq = frequencies[valid_range][np.argmax(fft_magnitudes_db[valid_range])]


    return frequencies, fft_magnitudes, fft_magnitudes_db, dominant_freq

def plot_spectrogram(audio_signal, sampling_rate):
    plt.figure(figsize=(10, 6))
    plt.specgram(audio_signal, Fs=sampling_rate, cmap="inferno")
    plt.title("Spectrogram")
    plt.xlabel("Waktu (detik)")
    plt.ylabel("Frekuensi (Hz)")
    plt.colorbar(label="Intensitas (dB)")
    plt.show()

def plot_frequency_spectrum(frequencies, magnitudes, magnitudes_db):
    plt.figure(figsize=(12, 6))
    
    # Plot linear scale
    plt.subplot(2, 1, 1)
    plt.plot(frequencies, magnitudes, color="blue")
    plt.title("Spektrum Frekuensi (Linear)")
    plt.xlabel("Frekuensi (Hz)")
    plt.ylabel("Amplitudo")
    plt.xlim(0, 5000) 
    
    # Plot log scale (dB)
    plt.subplot(2, 1, 2)
    plt.plot(frequencies, magnitudes_db, color="red")
    plt.title("Spektrum Frekuensi (Log dB)")
    plt.xlabel("Frekuensi (Hz)")
    plt.ylabel("Amplitudo (dB)")
    plt.xlim(0, 5000)  
    
    plt.tight_layout()
    plt.show()

# Eksekusi proses
audio_signal = record_audio(DURATION, SAMPLING_RATE)
frequencies, magnitudes, magnitudes_db, dominant_freq = compute_fft(audio_signal, SAMPLING_RATE)

# Cetak hasil frekuensi dominan
print(f"Frekuensi dominan yang terdeteksi: {dominant_freq:.2f} Hz")

# Plot hasil
plot_spectrogram(audio_signal, SAMPLING_RATE)
plot_frequency_spectrum(frequencies, magnitudes, magnitudes_db)