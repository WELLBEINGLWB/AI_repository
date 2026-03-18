% time_varying_pdnn.m
% MATLAB translation of time_varying_pdnn.py
%
% Time-Varying Certified Primal-Dual Neural Network (PDNN) solver
% for time-varying quadratic programming (TVQP).
%
% Implements the projected gradient PDNN iteration:
%   y_candidate = y - alpha * gamma * (H*y + p)
%   y_new       = P_Omega(y_candidate)   % box projection via min/max
%
% KKT vector:  y = [x (3x1); lambda (2x1); mu (1x1)]
%   x      - primal decision variables (box-constrained)
%   lambda - equality constraint dual variables (unbounded)
%   mu     - inequality constraint dual variable (non-negative)
%
% Reference: 2026 IEEE TAC - EIQP (Section VI TVQP example, Fig. 3)

clear; clc;

%% Parameters
gamma          = 5000;   % PDNN scaling factor (Fig. 3(c))
epsilon        = 1e-5;   % Convergence tolerance for inner loop
max_inner_iter = 50;     % Maximum inner PDNN iterations per time step
omega          = 1.0;    % Frequency parameter (Fig. 3 setup)
dt             = 0.001;  % Internal step size
UNBOUNDED_LIM  = 1e6;    % Effectively unbounded limit for dual variable bounds

T_END    = 10;
N_POINTS = 1000;

%% Time span and result storage
t_span    = linspace(0, T_END, N_POINTS);
y_current = ones(6, 1);           % Initial state: n=3 primal, m=2 eq., k=1 ineq.
results   = zeros(N_POINTS, 3);   % Each row stores primal x(t) at that time step

%% Main simulation loop
for idx = 1:N_POINTS
    t = t_span(idx);
    y_current = solve_step(t, y_current, gamma, epsilon, ...
                           max_inner_iter, dt, omega, UNBOUNDED_LIM);
    results(idx, :) = y_current(1:3)';
end

fprintf('Simulation complete for t=[0, 10]s.\n');
fprintf('Final Solution x(10): [%.6f, %.6f, %.6f]\n', results(end, :));

%% ============================= Local Functions =============================

function [W, q, J, d, A, b, xi_min, xi_max] = get_matrices(t, omega)
% GET_MATRICES  Build time-varying TVQP coefficient matrices at time t.
%
%   Inputs:
%     t     - current simulation time
%     omega - frequency parameter
%
%   Outputs:
%     W      - 3x3 symmetric positive definite cost matrix
%     q      - 3x1 linear cost vector
%     J      - 2x3 equality constraint matrix  (J*x = d)
%     d      - 2x1 equality constraint RHS
%     A      - 1x3 inequality constraint matrix (A*x <= b)
%     b      - 1x1 inequality constraint RHS
%     xi_min - 3x1 lower bounds on primal variables
%     xi_max - 3x1 upper bounds on primal variables

    w = omega;

    W = [2*cos(w*t)+22,   cos(w*t)-2,       3*sin(w*t)+6;
         cos(w*t)-2,      2*cos(2*w*t)+12,  sin(w*t);
         3*sin(w*t)+6,    sin(w*t),          cos(3*w*t)+8];

    q = [sin(3*w*t); cos(3*w*t); -cos(2*w*t)];

    J = [2*sin(4*w*t),  cos(w*t),      sin(w*t)+4;
         0.5*cos(w*t),  0.5*sin(w*t),  sin(2*w*t)];

    d = [sin(2*w*t); -cos(4*w*t)];

    A = [0.5*cos(w*t)+2,  sin(w*t)+1,  sin(4*w*t)+1];

    b = 1.5*cos(w*t)+8;

    xi_min = [cos(4*w*t)-6; sin(w*t)-6; sin(w*t+2)-6];
    xi_max = [cos(4*w*t)+6; sin(w*t)+6; sin(w*t+2)+6];
end


function y = solve_step(t, y_prev, gamma, epsilon, max_inner_iter, dt, omega, UNBOUNDED_LIM)
% SOLVE_STEP  One time step of the projected-gradient PDNN iteration.
%
%   Builds the 6x6 KKT matrix H and 6x1 vector p from the TVQP data,
%   then iterates the projected gradient update until convergence or
%   max_inner_iter is reached.
%
%   KKT structure (block partition [3 | 2 | 1]):
%     H = [ W    -J'    A' ]
%         [ J     0      0 ]
%         [-A     0      0 ]
%
%   Inputs:
%     t              - current time
%     y_prev         - 6x1 KKT vector from the previous time step
%     gamma          - PDNN scaling factor
%     epsilon        - convergence tolerance
%     max_inner_iter - maximum inner iterations
%     dt             - step size (alpha)
%     omega          - frequency parameter
%     UNBOUNDED_LIM  - bound value for effectively unconstrained variables
%
%   Output:
%     y - updated 6x1 KKT vector

    [W, q, J, d, A, b, xi_min, xi_max] = get_matrices(t, omega);

    % Assemble KKT matrix H (6x6)
    H = [W,    -J',          A';
         J,     zeros(2,2),  zeros(2,1);
        -A,     zeros(1,2),  zeros(1,1)];

    % Assemble KKT vector p (6x1)
    p = [q; -d; b];

    % Box-constraint bounds for the projection set Omega
    lb = [xi_min;                     -UNBOUNDED_LIM * ones(2,1); zeros(1,1)         ];
    ub = [xi_max;                      UNBOUNDED_LIM * ones(2,1); UNBOUNDED_LIM * ones(1,1)];

    alpha = dt;
    y = y_prev;

    for iter = 1:max_inner_iter
        y_candidate = y - alpha * gamma * (H * y + p);

        % Project onto box constraints (equivalent to np.clip)
        y_new = min(max(y_candidate, lb), ub);

        % Early termination if converged
        if norm(y_new - y) < epsilon
            y = y_new;
            return;
        end
        y = y_new;
    end
end
