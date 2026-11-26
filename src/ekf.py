class EKF:
    def __init__(self):
        self.x = np.array([0.0, 0.0, 0.0])
        self.P = np.eye(3)
        self.Q = np.array([
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0, 0, 0.05]
])
        self.R = np.array([
            [0.5, 0],
            [0, 0.5]
])

    def predict(self, v, omega, dt):

        x, y, theta = self.x

        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt

        theta_new = theta + omega * dt

        self.x = np.array([x_new, y_new, theta_new])

        F = np.array([
            [1, 0, -v*np.sin(theta)*dt],
            [0, 1, v*np.cos(theta)*dt],
            [0, 0, 1]
        ])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, gps):

        H = np.array([
            [1, 0, 0],
            [0, 1, 0]
])

        z = gps
        y = z - H @ self.x

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y

        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

    def state(self):
        return self.x