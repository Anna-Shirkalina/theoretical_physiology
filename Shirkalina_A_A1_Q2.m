clear variables;
clc; %clear the command screen
clf; %clear the figures

% Set simulation parameters & initialize system
dur = 5;  % duration of simulated process in s
Dt = 0.01;  % time step for simulation
t = 0;  % initial time
wait = ceil(0.5 / Dt); %interval betw target jumps, measured in time steps
n_q = 2; %number of dimensions of q
q_min = [-1.5; 0]; %the interval for min arm angles
q_max = [1.5; 2.5]; %the interval for max arm angles
q = [0;0]; %inital arm positions
q_est = q; % the agent's estimate of q, initially it has the correct estimate
q_star = [0;0]; %initial target position
q_vel = [0;0]; %initial arm velocity
q_vel_est = q_vel; % the agent's estimate of velocity, initially it has the correct estimate
q_acc = [0;0]; %initial arm acceleration
psi = [2.06 0.65 0.56]; % the psi vector, stored as a horizontal vector
psi_est = [1 1 1].* psi; % the agent's estimate of psi
M = [(psi(1)+2*psi(2)*cos(q(2,1))) (psi(3)+psi(2)*cos(q(2,1))); (psi(3)+psi(2)*cos(q(2,1))) psi(3)];
M_est = [(psi_est(1)+2*psi_est(2)*cos(q_est(2,1))) (psi_est(3)+psi_est(2)*cos(q_est(2,1))); (psi_est(3)+psi_est(2)*cos(q_est(2,1))) psi_est(3)];
GAMMA = psi(2)*sin(q(2,1))*[- q_vel(2,1) -(q_vel(2,1)+q_vel(1,1)); q_vel(1,1) 0];   
GAMMA_est = psi_est(2)*sin(q_est(2,1))*[- q_vel_est(2,1) -(q_vel_est(2,1)+q_vel_est(1,1)); q_vel_est(1,1) 0];   
step = 0;  % # time steps since simulation began
a = [0; 0]; %the initial action is zero

DATA = zeros(11, 1 + floor(dur/Dt));  % allocate at least enough memory
i_gp = 1;  % index of graph pts
DATA(:, i_gp) = [t; q_star; q; q_vel; q_acc; a];  % record data for plotting

for t = Dt:Dt:dur
    
  % Set target
  step = step + 1;
  if mod(step, wait) == 1
    q_star = q_min + (q_max - q_min).*rand(n_q, 1); %reset target
    q_est = q; %reset estimate to correct position
    q_vel_est = q_vel; %reset estimate to correct velocity
  end
  
%Update the M and Gamma matricies based on state positions   
M = [(psi(1)+2*psi(2)*cos(q(2,1))) (psi(3)+psi(2)*cos(q(2,1))); (psi(3)+psi(2)*cos(q(2,1))) psi(3)];
M_est = [(psi_est(1)+2*psi_est(2)*cos(q_est(2,1))) (psi_est(3)+psi_est(2)*cos(q_est(2,1))); (psi_est(3)+psi_est(2)*cos(q_est(2,1))) psi_est(3)];
GAMMA = psi(2)*sin(q(2,1))*[- q_vel(2,1) -(q_vel(2,1)+q_vel(1,1)); q_vel(1,1) 0];   
GAMMA_est = psi_est(2)*sin(q_est(2,1))*[- q_vel_est(2,1) -(q_vel_est(2,1)+q_vel_est(1,1)); q_vel_est(1,1) 0];   

  % Compute command
  h = - 40 *q_vel_est - 400 * (q_est - q_star); %my desired linear dynamics, based off the hurwitz polynomial
  a = M_est*h + GAMMA_est*q_vel_est;
  %a = min(4000, a) %optional caping of a below 4000, it doesn't usually
  %spike that high but just in case we can use this
  q_acc = inv(M)*(a - GAMMA*q_vel); %real state dynamics govern what our real acceleration will be 
  q_acc_est = inv(M_est)*(a - GAMMA_est*q_vel_est); %what the agent think's it's acceleration is, and it uses this to update it's position/velocity
  q_vel_est = q_vel_est + Dt * q_acc_est;
  q_est = q_est + Dt * q_vel_est;
  
  % Update arm position
  q_vel = q_vel + Dt*q_acc;   % Euler integration
  q = q + Dt*q_vel;  % Euler integration
  
  %make sure the arm coordinates are withing physiological norms
  q_bounded = max(q_min, min(q_max, q));
  q_vel = (q == q_bounded).*q_vel;
  q = q_bounded;
    
  % Record data for plotting
  i_gp = i_gp + 1;
  DATA(:, i_gp) = [t; q_star; q; q_vel; q_acc; a];

end  % for t
DATA = DATA(:, 1:i_gp);



% Plot
figure(1);
set(gcf, 'Name', 'Arm_positions and action graphs', 'NumberTitle', 'off');
subplot(2, 1, 1);
plot(DATA(1, :), DATA(2, :), 'r:');
hold on;
%grid on;
plot(DATA(1, :), DATA(3, :), 'b:');
plot(DATA(1, :), DATA(4, :), 'r');
plot(DATA(1, :), DATA(5, :), 'b');
ylim(1.05*[min(q_min), max(q_max)]);
ylabel('q');
set(gca, 'TickLength', [0, 0]);
subplot(2, 1, 2);
plot(DATA(1, :), DATA(10, :), 'b');
hold on;
plot(DATA(1, :), DATA(11, :), 'r');
hold on;
% line([0, dur], [a_max, a_max], 'Color', 'k', 'LineStyle', '--');
% line([0, dur], [-a_max, -a_max], 'Color', 'k', 'LineStyle', '--');
% ylim([-1.1*a_max, 1.1*a_max]);
ylabel('action');
xlabel('t');
set(gca, 'TickLength', [0, 0]);