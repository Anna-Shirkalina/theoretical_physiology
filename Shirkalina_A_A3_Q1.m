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
eta_Q = 1e-3;
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
Q_est = create_net([n_s + n_a; 20; 20; 1], 10*[0; 1; 1; 1], "relu");
mu_tgt = mu;
Q_tgt = Q_est;

% Set up graphic
figure(1);
set(gcf, 'Name', 'DDPG', 'NumberTitle', 'off');
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
    Q_est = forward_relu(Q_est, [s_; a_]);
    mu_tgt = forward_relu(mu_tgt, s_next_);
    a_next_ = mu_tgt.y{end} + a_sd*randn(n_a, 1); 
    Q_tgt = forward_relu(Q_tgt, [s_next_; a_next_]);
    Q_e = Q_est.y{end} - Dt*r_ - Q_tgt.y{end};  % Bellman error
    Q_est = backprop_relu_adam(Q_est, Q_e, eta_Q, adam_1, adam_2); 
        
    % Adjust actor (i.e. policy, mu) to minimize Q over buffer-minibatch 
    mu = forward_relu(mu, s_);
    a__ = mu.y{end};  % not usually = a_ from the buffer
    Q_est = forward_relu(Q_est, [s_; a__]);  % prepare for d_dx_relu
    dQ = d_dx_relu(Q_est, ones(1, n_m));
    dQ_da = dQ(n_s + 1:end, :);
    mu = backprop_relu_adam(mu, -dQ_da, eta_mu, adam_1, adam_2);
               
    % Nudge target nets toward learning ones
    for l = 2:Q_est.n_layers
      Q_tgt.W{l} = Q_tgt.W{l} + tau*(Q_est.W{l} - Q_tgt.W{l});
      Q_tgt.b{l} = Q_tgt.b{l} + tau*(Q_est.b{l} - Q_tgt.b{l});
    end
    for l = 2:mu.n_layers
      mu_tgt.W{l} = mu_tgt.W{l} + tau*(mu.W{l} - mu_tgt.W{l});
      mu_tgt.b{l} = mu_tgt.b{l} + tau*(mu.b{l} - mu_tgt.b{l});
    end
    
    % Update s
    s = s_next;

  end  % for t
  
  % Test policy
  if mod(rollout, 100) == 0
    R = test_policy(mu, s_test);
    Q_nse = batch_nse(Dt*r_ + Q_tgt.y{end}, Q_e);  % assess Q_est by computing its normalized squared error
    fprintf('At rollout %d, R = %.3f, Q_nse = %.4f\n', rollout, R, Q_nse)
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