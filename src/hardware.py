import numpy as np

class Probe2D:
    def __init__(self, nx=8, ny=8, pitch_x=0.6, pitch_y=0.6, gap_x=0.05, gap_y=0.05, freq_mhz=5.0, **kwargs):
        self.nx, self.ny = nx, ny
        self.pitch_x, self.pitch_y = pitch_x, pitch_y
        self.freq_mhz = freq_mhz
        # Les Gaps sont maintenant dynamiques (contrôlés par l'interface)
        self.gap_x = gap_x
        self.gap_y = gap_y
        
        # Calcul NDT exact de l'ouverture (Aperture)
        self.lx = (nx * pitch_x) - self.gap_x
        self.ly = (ny * pitch_y) - self.gap_y
        
        self.elements = self._generate_elements()

    def _generate_elements(self):
        x = np.linspace(-(self.lx-self.pitch_x)/2, (self.lx-self.pitch_x)/2, self.nx)
        y = np.linspace(-(self.ly-self.pitch_y)/2, (self.ly-self.pitch_y)/2, self.ny)
        xv, yv = np.meshgrid(x, y)
        return np.stack((xv.flatten(), yv.flatten()), axis=1)

class Wedge:
    def __init__(self, velocity=2330.0, angle_deg=36.0):
        self.velocity = velocity
        self.angle_deg = angle_deg

class Specimen:
    def __init__(self, velocity=5900.0):
        self.velocity = velocity