import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import sounddevice as sd

def record_audio(duration, sample_rate):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")
    return audio_data.flatten()

def find_fundamental_frequency(signal_data, sample_rate):
    # Hitung FFT
    N = len(signal_data)
    fft_values = np.fft.fft(signal_data)
    fft_magnitudes = np.abs(fft_values[:N // 2])  
    freqs = np.fft.fftfreq(N, d=1/sample_rate)[:N // 2] 
    

    peak_index = np.argmax(fft_magnitudes[1:]) + 1
    fundamental_freq = freqs[peak_index]
    
    return fundamental_freq, freqs, fft_magnitudes


duration = 3  
sample_rate = 44100 
signal_data = record_audio(duration, sample_rate)

fundamental_freq, freqs, fft_magnitudes = find_fundamental_frequency(signal_data, sample_rate)
print(f'Fundamental Frequency: {fundamental_freq:.2f} Hz')

# Plot hasil FFT
plt.plot(freqs, fft_magnitudes)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT Spectrum')
plt.grid()
plt.show()