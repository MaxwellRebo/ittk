import numpy as np
from ittk import mutual_information


def test_mutual_information():
    x = np.array([7, 7, 7, 3])
    y = np.array([0, 1, 2, 3])
    mut_inf = mutual_information(x, y)
    assert mut_inf == 0.8112781244591329
    x2 = [1, 0, 1, 1, 0]
    y2 = [1, 1, 1, 0, 0]
    mut_inf_two = mutual_information(x2, y2)
    assert mut_inf_two == 0.01997309402197492
