import numpy as np

class Probe2D:
    """Modélisation de la sonde matricielle (ex: IMASONIC 8x8)"""
    def __init__(self, nx=8, ny=8, pitch_x=0.6, pitch_y=0.6):
        self.nx = nx
        self.ny = ny
        self.pitch_x = pitch_x
        self.pitch_y = pitch_y
        self.elements = self._generate_coordinates()

    def _generate_coordinates(self):
        """Calcule la position (x, y, z) de chaque élément par rapport au centre de la sonde"""
        coords = []
        for i in range(self.nx):
            for j in range(self.ny):
                x = (i - (self.nx - 1) / 2) * self.pitch_x
                y = (j - (self.ny - 1) / 2) * self.pitch_y
                coords.append((x, y, 0.0)) # z=0 au niveau de la face émettrice
        return np.array(coords)

class Wedge:
    """Modélisation du sabot (ex: Rexolite)"""
    def __init__(self, velocity=2330.0, squat_angle_deg=36.0, roof_angle_deg=0.0):
        self.velocity = velocity # mm/s
        self.squat_angle = np.radians(squat_angle_deg)
        self.roof_angle = np.radians(roof_angle_deg)

class Specimen:
    """Modélisation de la pièce à inspecter"""
    def __init__(self, velocity=3240.0, thickness=50.0):
        self.velocity = velocity # mm/s (ex: Ondes Transversales Acier)
        self.thickness = thickness
