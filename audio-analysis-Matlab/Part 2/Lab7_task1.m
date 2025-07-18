% LAB7_TASK1: Modulate a speech waveform and plot amplitude spectra
% Load a speech waveform from a file and store it in the variable x
% The sample frequency is stored as Fs
[x, Fs] = audioread('lab07_speech0.wav');

% Transpose x to a row vector
x = x(:)';

% Carrier frequency in Hz
freq_carrier = 20e3;

% Cutoff frequency in Hz for the low pass filter
freq_cutoff = 4e3;

% Apply lowpass filter to message
x = lowpass(x, freq_cutoff, Fs);

% Modulate the message
xm = modulate(x, Fs, freq_carrier);

% Create vector of sample times
Ts = 1 / Fs; % sample period
Tmax = (length(x) - 1) * Ts;
t = 0:Ts:Tmax;

% Plot amplitude spectra of signals
figure(1); clf;
subplot(2, 1, 1);
plotAmplitudeSpectrum(x, Fs, 'Amplitude Spectrum of Message');
subplot(2, 1, 2);
plotAmplitudeSpectrum(xm, Fs, 'Amplitude Spectrum of Modulated Signal');