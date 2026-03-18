import numpy as np

# Represents an effectively unbounded limit for dual variable box constraints
_UNBOUNDED_LIMIT = 1e6


class TimeVaryingCertifiedPDNN:
    def __init__(self, gamma=1e4, epsilon=1e-5, max_inner_iter=50):
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_inner_iter = max_inner_iter
        self.omega = 1.0  # From Fig. 3 setup
        self.dt = 0.001   # Time step for simulation

    def get_matrices(self, t):
        w = self.omega
        # Simplified reconstruction of the Section VI TVQP coefficients
        W = np.array([
            [2*np.cos(w*t)+22, np.cos(w*t)-2, 3*np.sin(w*t)+6],
            [np.cos(w*t)-2, 2*np.cos(2*w*t)+12, np.sin(w*t)],
            [3*np.sin(w*t)+6, np.sin(w*t), np.cos(3*w*t)+8]
        ])
        q = np.array([np.sin(3*w*t), np.cos(3*w*t), -np.cos(2*w*t)])
        J = np.array([[2*np.sin(4*w*t), np.cos(w*t), np.sin(w*t)+4],
                      [0.5*np.cos(w*t), 0.5*np.sin(w*t), np.sin(2*w*t)]])
        d = np.array([np.sin(2*w*t), -np.cos(4*w*t)])
        A = np.array([[0.5*np.cos(w*t)+2, np.sin(w*t)+1, np.sin(4*w*t)+1]])
        b = np.array([1.5*np.cos(w*t)+8])
        xi_min = np.array([np.cos(4*w*t)-6, np.sin(w*t)-6, np.sin(w*t+2)-6])
        xi_max = np.array([np.cos(4*w*t)+6, np.sin(w*t)+6, np.sin(w*t+2)+6])
        return W, q, J, d, A, b, xi_min, xi_max

    def solve_step(self, t, y_prev):
        W, q, J, d, A, b, xi_min, xi_max = self.get_matrices(t)

        # Build H and p as per Eq. 6 (KKT system)
        # W: 3x3, J: 2x3, A: 1x3
        # y = [x (3), lambda (2), mu (1)]
        # Bug 1 fixed: zero blocks use correct semantic dimensions
        H = np.block([
            [W,  -J.T,              A.T             ],
            [J,   np.zeros((2, 2)), np.zeros((2, 1))],
            [-A,  np.zeros((1, 2)), np.zeros((1, 1))]
        ])
        p = np.concatenate([q, -d, b])

        # Projection bounds for Set Omega
        lb = np.concatenate([xi_min, -_UNBOUNDED_LIMIT*np.ones(2), np.zeros(1)])
        ub = np.concatenate([xi_max,  _UNBOUNDED_LIMIT*np.ones(2), _UNBOUNDED_LIMIT*np.ones(1)])

        # Bug 2, 3, 4 fixed: standard projected gradient update using gamma as
        # scaling factor; the M = I + H.T preconditioner has been removed.
        # The PDNN iteration solves: dy/dt = gamma * (P_Omega(y - (Hy+p)) - y)
        alpha = self.dt  # Internal step size matches configured dt
        y = y_prev.copy()

        for _ in range(self.max_inner_iter):
            y_candidate = y - alpha * self.gamma * (H @ y + p)
            y_new = np.clip(y_candidate, lb, ub)

            # Bug 5 fixed: use epsilon for early termination
            if np.linalg.norm(y_new - y) < self.epsilon:
                y = y_new
                break
            y = y_new

        return y


# --- Main Simulation Loop ---
# gamma=5000 as per Fig 3(c); T_END and N_POINTS control the time grid
T_END = 10
N_POINTS = 1000  # At least 1000 points for sufficient time resolution

solver = TimeVaryingCertifiedPDNN(gamma=5000)

# Bug 6 fixed: use at least 1000 points for sufficient time resolution
t_span = np.linspace(0, T_END, N_POINTS)

y_current = np.ones(6)  # Initial state: n=3, m=2, k=1
results = []

for t in t_span:
    y_current = solver.solve_step(t, y_current)
    results.append(y_current[:3])  # Extract primal x(t)

print(f"Simulation complete for t=[0, 10]s.")
print(f"Final Solution x(10): {results[-1]}")
