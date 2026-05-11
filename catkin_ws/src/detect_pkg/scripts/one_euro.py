import numpy as np

class LPF:
    def __init__(self):
        self._initialized = False
        self._prev = None

    def __call__(self, value: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = value
            return value
        filtered = alpha * value + (1.0 - alpha) * self._prev
        self._prev = filtered
        return filtered
    
    def reset(self):
        self._prev = None
        self._initialized = False

    @property
    def previous(self):
        return self._prev

class OneEuroFilter:
    def __init__(self, 
                 freq: float = 30.0, 
                 min_cutoff: float = 1.0, 
                 beta: float = 0.007,
                 dcutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._last_time = None
        self._lpf_x = LPF()
        self._lpf_dx = LPF()

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, value: np.ndarray, timestamp: float) -> np.ndarray:
        value = np.asarray(value)
        
        if self._last_time is not None and timestamp < self._last_time:
            # 时间回退，重置滤波器状态
            self._lpf_x.reset()
            self._lpf_dx.reset()
            self._last_time = None

        if self._last_time is None:
            dt = 1.0 / self.freq
        else:
            dt = timestamp - self._last_time
            if dt <= 0:
                dt = 1.0 / self.freq

        self._last_time = timestamp

        # 计算变化率 dx (速度)
        if self._lpf_x.previous is None:
            dx = np.zeros_like(value)
        else:
            dx = (value - self._lpf_x.previous) / dt

        # 1. 对速度 dx 进行低通滤波
        alpha_dx = self._alpha(self.dcutoff, dt)
        edx = self._lpf_dx(dx, alpha_dx)

        # 2. 根据平滑后的速度计算每个分量自适应的 cutoff 频率
        # np.abs(edx) 使得对于边界框 [cx, cy, w, h] 等各维度的滤波强度各自独立自适应
        cutoff = self.min_cutoff + self.beta * np.abs(edx)

        # 3. 计算自适应 alpha 并平滑原始值
        alpha_x = self._alpha(cutoff, dt)
        x_hat = self._lpf_x(value, alpha_x)

        return x_hat

    def reset(self):
        self._lpf_x.reset()
        self._lpf_dx.reset()
        self._last_time = None
