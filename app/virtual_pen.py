import math
import time
from typing import Tuple, Optional


class OneEuroFilter:
    def __init__(self, t0: float, x0: float, dx0: float = 0.0, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def _alpha(self, t: float, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / t)

    def __call__(self, t: float, x: float) -> float:
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
            
        alpha_d = self._alpha(dt, self.d_cutoff)
        dx = (x - self.x_prev) / dt
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(dt, cutoff)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat


class VirtualPen:
    """
    Advanced stabilization layer for handwriting.
    Implements a dead zone, One Euro filter, and predictive smoothing.
    """
    def __init__(self) -> None:
        self.filter_x: Optional[OneEuroFilter] = None
        self.filter_y: Optional[OneEuroFilter] = None
        
        self.dead_zone: float = 2.0
        self.prediction_ms: float = 0.015  # Predict 15ms ahead for responsiveness
        
        self.last_raw_x: Optional[float] = None
        self.last_raw_y: Optional[float] = None
        
        self.min_cutoff = 0.5  # Heavy smoothing at low speed
        self.beta = 0.01      # Fast reaction at high speed
        self.d_cutoff = 1.0
        
    def process(self, raw_pos: Tuple[int, int]) -> Tuple[int, int]:
        t = time.time()
        raw_x, raw_y = float(raw_pos[0]), float(raw_pos[1])
        
        if self.filter_x is None or self.filter_y is None or self.last_raw_x is None or self.last_raw_y is None:
            self.filter_x = OneEuroFilter(t, raw_x, min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
            self.filter_y = OneEuroFilter(t, raw_y, min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
            self.last_raw_x = raw_x
            self.last_raw_y = raw_y
            return (int(raw_x), int(raw_y))
            
        # Dead zone: discard micro-movements to kill stationary jitter
        dx = raw_x - self.last_raw_x
        dy = raw_y - self.last_raw_y
        if math.hypot(dx, dy) < self.dead_zone:
            raw_x = self.last_raw_x
            raw_y = self.last_raw_y
        else:
            self.last_raw_x = raw_x
            self.last_raw_y = raw_y
            
        # Apply One Euro Filter
        filtered_x = self.filter_x(t, raw_x)
        filtered_y = self.filter_y(t, raw_y)
        
        # Predictive smoothing (extrapolate slightly along current velocity to combat lag)
        vx = self.filter_x.dx_prev
        vy = self.filter_y.dx_prev
        
        pred_x = filtered_x + vx * self.prediction_ms
        pred_y = filtered_y + vy * self.prediction_ms
        
        return (int(pred_x), int(pred_y))
        
    def reset(self) -> None:
        self.filter_x = None
        self.filter_y = None
        self.last_raw_x = None
        self.last_raw_y = None
