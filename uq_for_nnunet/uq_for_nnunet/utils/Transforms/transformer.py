import numpy as np

class Transformer:
    """
    Transformer that:
    - Rotates changes axes from (z, x, y) to (x, y, z)
    - rotates 90 degress
    - flips up down
    """

    def transform_uncertainty_map(self, uq_array):
        uq_array = np.transpose(uq_array, (1,2,0))
        uq_array = np.rot90(uq_array, k=1)
        uq_array = np.flipud(uq_array)
        return uq_array
