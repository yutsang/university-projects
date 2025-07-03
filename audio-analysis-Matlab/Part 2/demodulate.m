function [rx, xd] = demodulate(x, Fs, freq_carrier, freq_cutoff)
% DEMODULATE: Demodulate a waveform X using a cosine carrier and lowpass filter.
%
%   [RX, XD] = DEMODULATE(X, FS, FREQ_CARRIER, FREQ_CUTOFF) demodulates a waveform
%   Inputs:
%       X            = waveform to demodulate (row or column vector)
%       FS           = sampling frequency (Hz)
%       FREQ_CARRIER = frequency of cosinusoidal carrier (Hz)
%       FREQ_CUTOFF  = cut-off frequency of low pass filter (Hz)
%   Outputs:
%       RX = demodulated waveform
%       XD = X mixed with cosinusoidal carrier wave (for debugging)

% Ensure x is a row vector
x = x(:)';

% Get the sample time and the duration of the signal
Ts = 1 / Fs; % sample period
Tmax = (length(x) - 1) * Ts;
t = 0:Ts:Tmax;

% Mix waveform with carrier (demodulation)
y = cos(2 * pi * freq_carrier * t);
xd = x .* y;

% Apply lowpass filtering
rx = lowpass(xd, freq_cutoff, Fs);

end