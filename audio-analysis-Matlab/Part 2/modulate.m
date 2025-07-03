function xm = modulate(x, Fs, freq_carrier)
% MODULATE: Modulate a cosinusoidal carrier wave with a message X.
%
%   XM = MODULATE(X, FS, FREQ_CARRIER) returns the modulated signal XM.
%   Inputs:
%       X            = the message signal (row or column vector)
%       FS           = the sampling frequency (Hz)
%       FREQ_CARRIER = the cosinusoidal carrier wave frequency (Hz)
%   Output:
%       XM = modulated signal

% Ensure x is a row vector
x = x(:)';

% Create vector of sample times
Ts = 1 / Fs; % sample period
Tmax = (length(x) - 1) * Ts;
t = 0:Ts:Tmax;

% Modulate the signal
carrier = cos(2 * pi * freq_carrier * t);
xm = x .* carrier;

end