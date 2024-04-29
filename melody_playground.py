import os
import numpy as np
import matplotlib.pyplot as plt
import torchaudio, torch
# read txt as numpy array
test_melody = "trimmed_30_Padding-S/ch/9/wav01/melody/wav01_003.txt"
# test_melody = "melody/mysinging.txt"

frequency = np.loadtxt(test_melody)
time = frequency[:, 0]
frequency = frequency[:, 1]

# Plot the melody
plt.figure(figsize=(10, 6))
i = 0
j = 1
while i < len(frequency) and j < len(frequency):
    if frequency[j] != 0 and j != len(frequency)-1:
        j += 1
    elif frequency[j] == 0 or j == len(frequency)-1:
        plt.plot(time[i:j-1], frequency[i:j-1], color='blue')
        i=j+1
        j+=1

plt.title('Melody Representation')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.grid(True)
plt.xlim(time.min(), time.max())

plt.savefig('melody_playground.png')

waveform, sample_rate = torchaudio.load("trimmed_30_Padding-S/ch/9/wav00/in/wav00_001.wav")

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.savefig('waveform_playground.png')

plot_waveform(waveform, sample_rate)