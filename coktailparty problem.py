import numpy as np
import wave
import warnings
from scipy.io import wavfile
import sounddevice as sd
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# Read the wave file

mix_1_wave = wave.open(r"C:\Users\Saiko-Store\Desktop\problem4\coktail\ICA mix 1.wav",'r')
mix_1_wave.getparams()

# Extract Raw Audio from Wav File
signal_1_raw = mix_1_wave.readframes(-1)
signal_1 = np.frombuffer(signal_1_raw, 'Int16')
'length: ', len(signal_1) , 'first 100 elements: ',signal_1[:100]


fs = mix_1_wave.getframerate()
timing = np.linspace(0, len(signal_1)/fs, num=len(signal_1))


plt.figure(figsize=(12,2))
plt.title('Recording 1')
plt.plot(timing,signal_1, c="#3ABFE7")
plt.show()
warnings.filterwarnings("ignore", message="The binary mode of fromstring is deprecated")

mix_2_wave = wave.open(r"C:\Users\Saiko-Store\Desktop\problem4\coktail\ICA mix 2.wav",'r')

#Extract Raw Audio from Wav File
signal_raw_2 = mix_2_wave.readframes(-1)
signal_2 = np.fromstring(signal_raw_2, 'Int16')


mix_3_wave = wave.open(r"C:\Users\Saiko-Store\Desktop\problem4\coktail\ICA mix 3.wav",'r')

#Extract Raw Audio from Wav File
signal_raw_3 = mix_3_wave.readframes(-1)
signal_3 = np.fromstring(signal_raw_3, 'Int16')


plt.figure(figsize=(12,2))
plt.title('Recording 2')
plt.plot(timing,signal_2, c="#3ABFE7")
plt.ylim(-35000, 35000)
plt.show()


plt.figure(figsize=(12,2))
plt.title('Recording 3')
plt.plot(timing,signal_3, c="#3ABFE7")
plt.ylim(-35000, 35000)
plt.show()
X = list(zip(signal_1, signal_2,signal_3))

# Let's peak at what X looks like
X[:10]
# Import FastICA

# Initialize FastICA with n_components=3
ica = FastICA(n_components=3)

# Run the FastICA algorithm using fit_transform on dataset X
ica_result = ica.fit_transform(X)
ica_result.shape
result_signal_1 = ica_result[:,0]
result_signal_2 = ica_result[:,1]
result_signal_3 = ica_result[:,2]
# Plot Independent Component #1
plt.figure(figsize=(12,2))
plt.title('Independent Component #1')
plt.plot(timing, result_signal_1)
plt.ylim(-0.010, 0.010)
plt.show()

# Plot Independent Component #2
plt.figure(figsize=(12,2))
plt.title('Independent Component #2')
plt.plot(timing, result_signal_2)
plt.ylim(-0.010, 0.010)
plt.show()

# Plot Independent Component #3
plt.figure(figsize=(12,2))
plt.title('Independent Component #3')
plt.plot(timing, result_signal_3)
plt.ylim(-0.010, 0.010)
plt.show()

 
# Convert to int, map the appropriate range, and increase the volume a little bit
result_signal_1_int = np.int16(result_signal_1*32767*100)
result_signal_2_int = np.int16(result_signal_2*32767*100)
result_signal_3_int = np.int16(result_signal_3*32767*100)
# play files
sd.play(result_signal_1_int, fs)
sd.play(result_signal_2_int, fs)
sd.play(result_signal_3_int, fs)
# Write wave files
wavfile.write("result1.wav", fs, result_signal_1_int)
wavfile.write("result2.wav", fs, result_signal_2_int)
wavfile.write("result3.wav", fs, result_signal_3_int)
