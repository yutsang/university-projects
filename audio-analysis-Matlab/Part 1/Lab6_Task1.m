% LAB6_TASK1: Analyze and approximate a speech frame
% Load a speech signal
[x, Fs] = audioread('test.wav');

% Define the frame length
frame_length = 512;

% Compute power of each frame
frame_power = get_frame_power(x, frame_length);

% Plot original waveform and power in each frame
figure(1); clf;
plot_speech_power(x, frame_power, frame_length);

% Find index of frame with maximum power (fnum)
[mxpow, fnum] = max(frame_power);

% Extract that frame
frame = extract_frame(x, frame_length, fnum);

% Approximate frame with 4 largest frequency components
numcom = 4;
app_frame = approximate_frame(frame, numcom);

% Compare the original and approximated frame
figure(2); clf;
plot_approx(frame, app_frame, numcom);