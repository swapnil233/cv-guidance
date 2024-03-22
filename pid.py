class PIDController:
    def __init__(
        self,
        kp,
        ki,
        kd,
        setpoint=0,
        integral_limit=None,
        derivative_filter_tau=0.0,
        setpoint_weights=(1, 1),
    ):
        """Initialize the PID controller with gains, setpoint, and enhancements."""
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint  # Desired setpoint

        self.integral_limit = (
            integral_limit  # Limit for the integral term (Anti-windup)
        )
        self.derivative_filter_tau = (
            derivative_filter_tau  # Time constant for derivative low-pass filter
        )
        self.setpoint_weights = (
            setpoint_weights  # Weights for setpoint in P and D terms
        )

        self.last_error = 0.0  # Last error value
        self.integral = 0.0  # Integral of error
        self.derivative_filtered = 0.0  # Filtered derivative

    def reset(self):
        """Reset the controller state."""
        self.last_error = 0.0
        self.integral = 0.0
        self.derivative_filtered = 0.0

    def update(self, current_value, dt):
        """Update the PID controller with enhancements."""
        # Calculate error
        error = self.setpoint - current_value

        # Proportional term with setpoint weighting
        P = self.kp * (self.setpoint_weights[0] * self.setpoint - current_value)

        # Integral term with anti-windup
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = max(
                min(self.integral, self.integral_limit), -self.integral_limit
            )
        I = self.ki * self.integral

        # Derivative term with filtering and setpoint weighting
        derivative_raw = (error - self.last_error) / dt
        self.derivative_filtered += (
            (derivative_raw - self.derivative_filtered)
            * dt
            / (self.derivative_filter_tau + dt)
            if self.derivative_filter_tau > 0
            else derivative_raw
        )
        D = self.kd * self.derivative_filtered * self.setpoint_weights[1]

        # Remember last error for next derivative calculation
        self.last_error = error

        # Calculate total control output
        control_action = P + I + D

        return control_action
