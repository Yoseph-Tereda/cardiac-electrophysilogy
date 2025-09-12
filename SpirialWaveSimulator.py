"""
summary
"""


import numpy as np


class SpirialWaveSimulator():
    """

    """

    def __init__(self,BBx,AAx,sigma_i,cm,chi,n_x,n_y,t_invt,b) :
        self.sigma_i = sigma_i
        self.cm = cm
        self.chi = chi
        self.n_x = n_x
        self.n_y = n_y
        self.t_invt = t_invt
        self.b = b
        self.BBx = BBx
        self.AAx = AAx

    def constant_matrix(self):
        """

        :return:
        """
        cons_mat = np.linalg.inv(np.kron(self.BBx, self.BBx)).dot((-(self.sigma_i/(self.cm*self.chi))*(np.kron(self.AAx, self.BBx)+ np.kron(self.BBx, self.AAx))))
        return cons_mat

    def trans_pot_ic(self):
        """

        :return:
        """
        v_ic = np.zeros((self.n_x, self.n_y)) # trans_pot_ic represents the initial condition for transmembrane potential.
        v_ic[0:24, 0:24] = 1
        tra_v_ic_column = np.array(np.reshape(v_ic, -1, order='F'))
        tra_v_ic_column = tra_v_ic_column.T
        return tra_v_ic_column

    def sta_var_cellmodel_ic(self):
        """

        :return:
        """
        s_ic = np.zeros((self.n_x, self.n_y)) # sta_var_cellmodel represents the initial condition for a vector of state variable from cell model.
        s_ic[0:24, 25:50] = 0.1
        s_ic[25:50, 25:50] = 0.1
        tra_s_ic_column = np.array(np.reshape(s_ic, -1, order='F'))
        tra_s_ic_column = tra_s_ic_column.T
        return tra_s_ic_column

    def step_size_FE(self):
        """

        :return:
        """
        egv= np.linalg.eigvalsh(self.constant_matrix())
        min_eigen_value = min(egv)
        step_size = -2/min_eigen_value
        return step_size

    def number_mesh_time(self):
        """

        :return:
        """
        total_mesh_point=round((self.t_invt[1]-self.t_invt[0])/self.step_size_FE()) # number_mesh_time represents the total number of mesh point on time interval.
        return total_mesh_point


