clc; close all; clear all;

addpath('D:/OneDrive - University of Florida/Research/myPapers/Paper4/casadi2')
import casadi.*

% Tunable Parameters
T = 0.2; % sampling time
N = 10; % prediction horizon
Q = diag([1;5;0.1]); % cost matrix for states
R = diag([0.5;0.05]); % cost matrix for inputs

vMax = 0.6; 
vMin = -vMax;
omegaMax = pi/4;
omegaMin = -omegaMax;

% Declare states
x = SX.sym('x');
y = SX.sym('y');
theta = SX.sym('theta');
states = [x;y;theta];
numStates = length(states);

% Declare control inputs
v = SX.sym('v');
omega = SX.sym('omega');
controls = [v;omega]; numControls = length(controls);
rhs = [v*cos(theta); v*sin(theta); omega]; % xDot equations / System RHS

% Declare problem structures
f = Function('f',{states,controls},{rhs}); % nonlinear mapping function xDot = f(x,u)
U = SX.sym('U',numControls, N); % Decision variables (controls)
P = SX.sym('P',numStates + numStates); % parameters that include initial state and desired state
X = SX.sym('X',numStates, N+1); % matrix that contains the states over the problem
obj = 0;
g = []; % Constraint vector

%% Construct constraints
st = X(:,1); %initial state
g = [g;st-P(1:numStates)];
for k = 1:N
    st = X(:,k); con = U(:,k);
    obj = obj + (st-P(numStates+1:2*numStates))'*Q*(st-P(numStates+1:2*numStates)) + con'*R*con; % compute objective
    stNext = X(:,k+1);
    % 4th order Runge-Kutta
    k1 = f(st, con);
    k2 = f(st + T/2*k1, con);
    k3 = f(st + T/2*k2, con);
    k4 = f(st + T*k3, con);

    stNextRK4 = st + T/6*(k1 + 2*k2 + 2*k3 + k4);
    g = [g;stNext-stNextRK4]; % Multiple shooting path continuity constraint
end

% Assemble optimization problem using defined structures
OPTvars = [reshape(X,numStates*(N+1),1); reshape(U,numControls*N,1)]; % make decision variables into column vector
nlp_prob = struct('f', obj, 'x', OPTvars, 'g', g, 'p', P);

opts = struct;
opts.ipopt.max_iter = 100;
opts.ipopt.print_level = 0;
opts.print_time = 0;
opts.ipopt.acceptable_tol = 1e-8; % 1e-6
opts.ipopt.acceptable_obj_change_tol = 1e-6;
solver = nlpsol('solver', 'ipopt', nlp_prob, opts);

% Define Constraints
args = struct;

    % Equality Constraints
    args.lbg(1:numStates*(N+1)) = 0; % State path equality constriants 
    args.ubg(1:numStates*(N+1)) = 0; % lbg = ubg for equality constraints
    
    % State limits
    args.ubx(1:numStates:numStates*(N+1),1) = 2; % state x upper bound
    args.ubx(2:numStates:numStates*(N+1),1) = 2; % state y upper bound
    args.ubx(3:numStates:numStates*(N+1),1) = inf; % state omega upper bound

    args.lbx(1:numStates:numStates*(N+1),1) = -2; % state x lower bound
    args.lbx(2:numStates:numStates*(N+1),1) = -2; % state y lower bound
    args.lbx(3:numStates:numStates*(N+1),1) = -inf; % state omega lower bound
    
    % input constraints    
    args.lbx(numStates*(N+1) + 1 : numControls : numStates*(N+1) + numControls*N,1) = vMin; % v lower bound
    args.lbx(numStates*(N+1) + 2 : numControls : numStates*(N+1) + numControls*N,1) = omegaMin; % omega lower bound
  
    args.ubx(numStates*(N+1) + 1 : numControls : numStates*(N+1) + numControls*N,1) = vMax; % v upper bound
    args.ubx(numStates*(N+1) + 2 : numControls : numStates*(N+1) + numControls*N,1) = omegaMax; % omega upper bound

%------------------------------%
%       SIMULATION LOOP        %
%------------------------------%
t0 = 0;
x0 = [0; 0; 0.0]; % initial state
xs = [1.5; 1.5; 0.0]; % desired state

xx(:,1) = x0; % xx contains the history of states
t(1) = t0;
u0 = zeros(N,numControls); % six control inputs
X0 = repmat(x0,1,N+1)'; % initilization of states decision variables

% Start MPC
iter = 0;
xx1 = []; % stores predicted state
u_cl = []; % stores control actions
tol = 1e-2;

while(norm((x0-xs),2) > tol)
    args.p = [x0;xs]; % set values of parameter vector
    args.x0 = [reshape(X0',numStates*(N+1),1); reshape(u0',numControls*N,1)]; % intial value of optimization variables
    tic;
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx, 'lbg', ...
        args.lbg, 'ubg', args.ubg, 'p', args.p);
    solutionTime(iter+1) = toc; % store execution time

    % Store solution
    u = reshape(full(sol.x(numStates*(N+1)+1:end))',numControls,N)'; % Reshaping u from a vector to a matrix (only get controls from the solution)
    xx1(:,1:numStates,iter+1) = reshape(full(sol.x(1:numStates*(N+1)))',numStates,N+1)'; % get solution trajectory

    % get solution trajectory
    u_cl = [u_cl; u(1,:)]; % store first control action from optimal sequence
    t(iter+1) = t0;

    % Apply control and shift solution
    [t0, x0, u0] = shift(T, t0, x0, u, f);
    xx(:,iter+2) = x0;
    X0 = reshape(full(sol.x(1:numStates*(N+1)))',numStates,N+1);

    % Shift trajectory to initilize next step
    X0 = [X0(2:end,:); X0(end,:)];
    iter = iter + 1


end

t = 0:T:T*iter;

figure
plot(xx(1,:),xx(2,:))
xlim([-0.1,2])
ylim([-0.1,2])



function [t0, x0, u0] = shift(T, t0, x0, u, f)
    st = x0;
    con = u(1,:)';

    k1 = f(st, con);
    k2 = f(st + T/2*k1, con);
    k3 = f(st + T/2*k2, con);
    k4 = f(st + T*k3, con);

    st = st + T/6*(k1 + 2*k2 + 2*k3 + k4);


    x0 = full(st);

    t0 = t0 + T;
    u0 = [u(2:size(u,1),:); u(size(u,1),:)];
end




































