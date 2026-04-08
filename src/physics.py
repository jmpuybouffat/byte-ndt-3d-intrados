import numpy as np
from scipy.optimize import minimize

class FocalLawCalculator:
    def __init__(self, probe, wedge, specimen):
        self.probe = probe
        self.wedge = wedge
        self.specimen = specimen

    def compute_fermat_3d(self, tx, ty, tz):
        v1, v2 = self.wedge.velocity, self.specimen.velocity
        target = np.array([tx, ty, tz])
        tof, points_i = [], []

        for el in self.probe.elements:
            z_el = -5.0 - (el[0] + self.probe.lx/2) * np.tan(np.radians(self.wedge.angle_deg))
            el_pos = np.array([el[0], el[1], z_el])
            
            def time_func(p):
                pi = np.array([p[0], p[1], 0.0])
                return (np.linalg.norm(pi - el_pos)/v1) + (np.linalg.norm(target - pi)/v2)
            
            res = minimize(time_func, [0, 0], method='Nelder-Mead')
            tof.append(res.fun)
            points_i.append([res.x[0], res.x[1], 0.0])
            
        tof = np.array(tof)
        # Conversion stricte en nanosecondes ENTIÈRES pour le FPGA
        delays_ns = np.round((np.max(tof) - tof) * 1e9).astype(int)
        return delays_ns, np.array(points_i)