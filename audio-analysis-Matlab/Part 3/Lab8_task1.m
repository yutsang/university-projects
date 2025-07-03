% LAB8_TASK1: Simulate ALOHA protocol and compute efficiency
rng(0); % Make the rand() generate exactly the same 'random' numbers

% Parameters
n_slots = 1000; % Total number of slots
p = 0.1;        % Probability a node transmits
n_users = 4;    % Number of nodes

% Initialize the simulation of the ALOHA system
initSimulation(n_users, n_slots);

% Result counters
n_succ = 0; % Number of successful transmissions
n_empty = 0; % Number of empty slots
n_coll = 0; % Number of slots with collisions
slot_status = zeros(n_users, n_slots); % Used for the plot

% Simulate the transmission process for n_slots
for t = 1:n_slots % Loop for each slot
    slot = zeros(1, 16); % Reset the slot
    for id = 1:n_users
        % Get the current frame and check if it is empty
        frame = getCurrentFrame(id);
        % Determine when to transmit
        if rand(1) < p % Generate a Bernoulli random variable with parameter p
            % Transmission: update the slot using a logical or
            slot = or(slot, getFrame(id));
            % Delete the frame, so that we can transmit a new one
            resetFrame(id);
            % Update the slot_status for the final plot
            slot_status(id, t) = 1;
        end
    end
    % Is there a new message?
    if ~isequal(slot, zeros(1, 16))
        % Check the received message
        if checkReceivedFrame(slot, n_users)
            n_succ = n_succ + 1; % Update the counter of frame transmitted successfully
        else
            n_coll = n_coll + 1; % Update the counter for the collisions
        end
    else
        n_empty = n_empty + 1; % Update the number of empty frames
    end
end

% Compute efficiency
% Efficiency = successful transmissions / total slots

% Avoid division by zero
if (n_succ + n_coll + n_empty) > 0
    efficiency = n_succ / (n_succ + n_coll + n_empty);
else
    efficiency = 0;
end

% Display results
fprintf('Total number of slots: %d\n', n_slots);
fprintf('Empty slots: %d\n', n_empty);
fprintf('Collisions: %d\n', n_coll);
fprintf('Frame transmitted successfully: %d\n', n_succ);
fprintf('Efficiency: %.4f\n', efficiency);

% Plot
fh = figure(1); clf;
window_size = 50;
plotTraffic(fh, slot_status, n_users, window_size);