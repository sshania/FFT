import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import get_window

def nextpow2(n):
    return int(np.ceil(np.log2(n)))

def detect_frequency_HPS(filepath, max_harmonic=5):

    sample_rate, data = wavfile.read(filepath)
    print(f"Loaded '{filepath}' dengan sample rate {sample_rate} Hz")

 
    if data.ndim > 1:
        data = data[:, 0]
    

    data = data / np.max(np.abs(data))
    
 
    windowed = data * get_window('blackman', len(data))
    

    N = len(windowed)
    N_fft = 2 ** nextpow2(N)
    spectrum = np.abs(np.fft.fft(windowed, n=N_fft))
    spectrum = spectrum[:N_fft//2] 
 
    hps_spectrum = spectrum.copy()
    for h in range(2, max_harmonic + 1):
        decimated = spectrum[::h]
        hps_spectrum[:len(decimated)] *= decimated


    peak_index = np.argmax(hps_spectrum)
    freqs = np.fft.fftfreq(N_fft, d=1/sample_rate)[:N_fft//2]
    dominant_freq = freqs[peak_index]

    # Plot
    plt.plot(freqs[:1000], hps_spectrum[:1000])  
    plt.xlabel("Frekuensi (Hz)")
    plt.ylabel("HPS Magnitude")
    plt.title("Harmonic Product Spectrum")
    plt.grid()
    plt.show()

    print(f"Frekuensi Dominan (HPS) from {sample_rate} : {dominant_freq:.2f} Hz")
    return dominant_freq

# Tes senar gitar
# detect_frequency_HPS("RekamanGuitar/Senar_Gitar_1.wav")
# detect_frequency_HPS("RekamanGuitar/Senar_Gitar_2.wav")
# detect_frequency_HPS("RekamanGuitar/Senar_Gitar_3.wav")
# detect_frequency_HPS("RekamanGuitar/Senar_Gitar_4.wav")
# detect_frequency_HPS("RekamanGuitar/Senar_Gitar_5.wav")
# detect_frequency_HPS("RekamanGuitar/Senar_Gitar_6.wav")

#electric
# detect_frequency_HPS("RekamanElectricGuitar/ES6.wav")
# detect_frequency_HPS("RekamanElectricGuitar/ES5.wav")
# detect_frequency_HPS("RekamanElectricGuitar/ES4.wav")
# detect_frequency_HPS("RekamanElectricGuitar/ES3.wav")
# detect_frequency_HPS("RekamanElectricGuitar/ES2.wav")
# detect_frequency_HPS("RekamanElectricGuitar/ES1.wav")

detect_frequency_HPS("RekamanViolin/G.wav")
detect_frequency_HPS("RekamanViolin/D.wav")
detect_frequency_HPS("RekamanViolin/A.wav")
detect_frequency_HPS("RekamanViolin/E.wav")