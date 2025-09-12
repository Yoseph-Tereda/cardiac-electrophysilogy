import numpy as np
import matplotlib.pyplot as plt
from SpirialWaveSimulator import SpirialWaveSimulator
from Data import BBx
from Data import AAx

if __name__ == '__main__':

    sigma_i = 0.1
    cm = 1
    chi = 1000
    n_x = 51
    n_y = 51
    t_invt = [0, 1000]
    b = 1
    scw = SpirialWaveSimulator(BBx,AAx,sigma_i, cm, chi, n_x, n_y, t_invt, b)

    print('I am here')

    #comment 1
    v = np.zeros((n_x * n_y, scw.number_mesh_time()))
    #comment 2
    v[:, 0] = scw.trans_pot_ic()
    #comment 3
    s = np.zeros((n_x * n_y, scw.number_mesh_time()))
    #comment 4
    s[:, 0] = scw.sta_var_cellmodel_ic()

    #comment 5
    dvvdtime = np.zeros((n_x * n_y, 1))
    dssdtime = np.zeros((n_x * n_y, 1))
    dt = scw.step_size_FE()

    print('I am here 2')

    #comment 6
    for n in range(0, scw.number_mesh_time() - 1):
        dvvdtime = v[:, n] * ((1 - v[:, n]) * (v[:, n] - 0.1)) - s[:, n]
        dssdtime = 0.01 * (0.5 * v[:, n] - s[:, n])
        s[:, n + 1] = s[:, n] + dt * (dssdtime * b)
        v[:, n + 1] = v[:, n] + dt * dvvdtime * b
        v[:, n + 1] = v[:, n + 1] + (scw.constant_matrix().dot(v[:, n + 1])) * b * dt


    print('I am here 3')

    plt.figure()
    plt.contourf(np.reshape(v[:, scw.number_mesh_time() - 1], [n_x, n_y]))
    plt.show()

    print("done")


