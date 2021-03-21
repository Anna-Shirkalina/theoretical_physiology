clear variables;
clc;
clf;

global n_q Dt n_steps r s_path a_path s_test q_min q_max s_star
Dt = 0.1;
n_q = 2;
n_s = 2*n_q;
n_r = 2;  % no. of elements of q that affect reward
n_qR = 2;  % no. of elements of q that affect return, must be >= n_r
n_a = n_qR;
i_qr = 1:n_r;
Dt = 0.1;
s_star = [0; 0.5; 0; 0]; %the constant s_star is used in the reward function
s_test = [-1; 1; 0; 0]; % Set initial states for test batch
q_min = [-1.5; 0];
q_max = [1.5; 2.5];

% Define task
r = @(s, a) -10*(s - s_star)'*(s - s_star) - 0.01*(a'*a);
dur = 3;
n_steps = floor(1 + dur/Dt);

% Set up learning
eta_V = 1e-3;
eta_f = 1e-3;
eta_r = 1e-3;
eta_mu = 1e-5; 
tau = 3e-4;
adam_1 = 0.9;
adam_2 = 0.999;
a_sd = 0.1;
n_rollouts = 2000;
n_m = 100;
n_buf = 10000;  % how many columns can the buffer hold?
buf_fill = 0;   % how many columns have been filled so far?
buf_i = 0;  % index of the most recent column to be filled 
BUFFER = zeros(2*n_s + n_a + 1, n_buf);




% Create nets
rng(25);
mu = create_net([n_s; 100; 100; n_a], 10*[0; 1; 1; 1], "relu");
V_est = create_net([n_s; 20; 20; 1], 10*[0; 1; 1; 1], "relu");
f_est = create_net([n_s + n_a; 100; 100; n_s], 10*[0; 1; 1; 1], "relu");
r_est = create_net([n_s + n_a; 20; 20; 1], 10*[0; 1; 1; 1], "relu");
V_tgt = V_est;
f_tgt = f_est;

% Set up graphic
figure(1);
set(gcf, 'Name', 'A3_Q2', 'NumberTitle', 'off');
s_path = zeros(n_s, n_steps);
a_path = zeros(n_a, n_steps);

% Assess initial policy
R = test_policy(mu, s_test);
fprintf('At rollout 0, R = %.3f\n', R);

% Train policy
for rollout = 1:n_rollouts

  % Run rollout
  s = 2*(rand(n_s, 1) - 0.5);
  q = max(s(1:2), q_min); %check that the bounds of q are valid
  q = min(q, q_max);
  s(1: 2, 1) = q;
  
  for t = 1:n_steps

    % Compute transition
    mu = forward_relu(mu, s);
    a = mu.y{end} + a_sd*randn(n_a, 1);  % exploration
    s_next = arm_dyn(s, a);

    % Store in the buffer
    BUFFER(:, buf_i + 1) = [s; a; r(s, a); s_next];
    buf_i = mod(buf_i + 1, n_buf);
    buf_fill = min(buf_fill + 1,n_buf);

    % Choose a minibatch from the buffer
    i = ceil(buf_fill*rand(1, n_m));
    s_ = BUFFER(1:n_s, i);
    a_ = BUFFER(n_s + 1:n_s + n_a, i);
    r_ = BUFFER(n_s + n_a + 1:n_s + n_a + 1, i);
    s_next_ = BUFFER(n_s + n_a + 2:end, i);
    
    % Adjust critic (i.e. Q_est) to minimize the squared Bellman error over the buffer-minibatch
    f_est = forward_relu(f_est, [s_; a_]);
    V_est = forward_relu(V_est, f_est.y{end});
    V_tgt = forward_relu(V_tgt, s_next_);
    r_est = forward_relu(r_est, [s_; a_]);
    % Bellman error
    %Q_e = Q_est.y{end} - Dt*r_ - V_tgt.y{end};  
    r_e = r_est.y{end} - r_;
    f_e = V_est.y{end} - forward_relu(V_est, s_next_).y{end};
    %Bellman error for V_e
    a_next_ = forward_relu(mu, s_next_).y{end};
    r_est_s1 = r_est.y{end};
    r_est_s2 = forward_relu(r_est, [s_next_; a_next_]).y{end};
    f_tgt = forward_relu(f_tgt, [s_next_; a_next_]);
    V_tgt = forward_relu(V_tgt, f_tgt.y{end});
    V_e = V_est.y{end} + Dt*r_est_s1 - Dt*r_ - V_tgt.y{end} - r_est_s2;
    %Backprops
    V_est = backprop_relu_adam(V_est, V_e, eta_V, adam_1, adam_2); 
    f_est = backprop_relu_adam(f_est, f_e.*[1;1;1;1], eta_f, adam_1, adam_2);
    r_est = backprop_relu_adam(r_est, r_e, eta_r, adam_1, adam_2);
        
    % Adjust actor (i.e. policy, mu) to minimize V over buffer-minibatch 
    mu = forward_relu(mu, s_);
    V_est = forward_relu(V_est, s_);  % prepare for d_dx_relu
    dV_ds = d_dx_relu(V_est, ones(1, n_m));
    da_ds_ = d_dx_relu(mu, ones(2, n_m)); %assuming mu is one-to-one (big potentially unjustified assumption) then ds_da =  1 / da_ds
    dV_da = dot(dV_ds, (1./da_ds_));
    mu = backprop_relu_adam(mu, -dV_da.*[1;1], eta_mu, adam_1, adam_2);
               
    % Nudge target nets toward learning ones
    for l = 2:f_est.n_layers
      f_tgt.W{l} = f_tgt.W{l} + tau*(f_est.W{l} - f_tgt.W{l});
      f_tgt.b{l} = f_tgt.b{l} + tau*(f_est.b{l} - f_tgt.b{l});
    end
    for l = 2:V_est.n_layers
      V_tgt.W{l} = V_tgt.W{l} + tau*(V_est.W{l} - V_tgt.W{l});
      V_tgt.b{l} = V_tgt.b{l} + tau*(V_est.b{l} - V_tgt.b{l});
    end
    % Update s
    s = s_next;

  end  % for t
  
  % Test policy
  if mod(rollout, 100) == 0
    R = test_policy(mu, s_test);
    V_nse = batch_nse(Dt*r_ + V_tgt.y{end}, V_e);  % assess Q_est by computing its normalized squared error
    f_nse = batch_nse(f_tgt.y{end}, f_e);
    r_nse = batch_nse(r_est.y{end}, r_e);
    fprintf('At rollout %d, R = %.3f, V_nse = %.4f, f_nse = %.4f, r_nse = %.4f\n', rollout, R, V_nse, f_nse, r_nse)
  end

end  % for rollout


function s_next = arm_dyn(s, a)
global Dt q_min q_max
psi = [2.06 0.65 0.56]; % the psi vector, stored as a horizontal vector
q = [s(1:2,1)]; %the state vector is stored as a 1x4 row vector, so need to extract the postions and velocity row vectors
q_vel = [s(3:4,1)];
M = [(psi(1)+2*psi(2)*cos(q(2,1))) (psi(3)+psi(2)*cos(q(2,1))); (psi(3)+psi(2)*cos(q(2,1))) psi(3)];
GAMMA = psi(2)*sin(q(2,1))*[-q_vel(2,1) -(q_vel(2,1)+q_vel(1,1)); q_vel(1,1) 0]; 


q_acc = inv(M)*(a - GAMMA*q_vel);

q = q + Dt*q_vel;
q_vel = q_vel + Dt*q_acc;   % Euler integration
q = max(q, q_min); %check that the bounds of q are valid
q = min(q, q_max);

s_next = [q; q_vel];
end

function R = test_policy(mu, s)

  global n_q Dt n_steps r s_path a_path s_star
  
  R = 0;
  for t = 1:n_steps
    mu = forward_relu(mu, s);
    a = mu.y{end};
    R = R + r(s, a);
    s_path(:, t) = s(:, 1);
    a_path(:, t) = a(:, 1);
    s = arm_dyn(s, a);
  end  % for t
  R = R*Dt;
    
  % Plot q & a for one test movement
  a_plot = a_path; 
  q_plot = s_path(1:n_q, :);
  subplot(2, 1, 1);
  plot(1:n_steps, q_plot);
  ylim([-1.5, 2.5]);
  grid on;
  ylabel('q');
  set(gca, 'TickLength', [0, 0])
  subplot(2, 1, 2); 
  plot(1:n_steps, a_plot);
  ylim([-1.1, 1.1]);
  grid on;
  ylabel('action');
  xlabel('t');
  set(gca, 'TickLength', [0, 0]);
  drawnow;

end  % function test_policy



