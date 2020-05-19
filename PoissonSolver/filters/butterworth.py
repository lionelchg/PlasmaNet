import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

fig_dir = 'figures/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

if __name__ == '__main__':

    b, a = signal.butter(10, 100, 'low', analog=True)
    w, h = signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green') # cutoff frequency
    plt.savefig(fig_dir + 'butterworth_lp_bode', bbox_inches='tight')

    t = np.linspace(0, 1, 1000, False)  # 1 second
    sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    ax1.plot(t, sig)
    ax1.set_title('10 Hz and 20 Hz sinusoids')
    ax1.axis([0, 1, -2, 2])

    sos_high = signal.butter(10, 15, 'hp', fs=1000, output='sos')
    filtered_high = signal.sosfilt(sos_high, sig)
    ax2.plot(t, filtered_high)
    ax2.set_title('After 15 Hz high-pass filter')
    ax2.axis([0, 1, -2, 2])
    ax2.set_xlabel('Time [seconds]')
    plt.tight_layout()
    plt.savefig(fig_dir + 'high_pass_filter', bbox_inches='tight')

    sos_low = signal.butter(10, 12, 'lp', fs=1000, output='sos')
    filtered_low = signal.sosfilt(sos_low, sig)
    ax3.plot(t, filtered_low)
    ax3.set_title('After 12 Hz low-pass filter')
    ax3.axis([0, 1, -2, 2])
    ax3.set_xlabel('Time [seconds]')
    plt.tight_layout()
    plt.savefig(fig_dir + 'filtered_signal', bbox_inches='tight')