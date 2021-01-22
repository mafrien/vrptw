import math

import numpy as np
from scipy.integrate import quad


class Spatiotemporal:
    def __init__(self, dataset, tws, service_time,
                 k1, k2, k3, alpha1, alpha2):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.tws_all = tws
        self.MAX_TW = self.get_max_tw()

        self.service_time_all = service_time

        self.points_all = dataset

        length = len(self.points_all)
        self.euclidian_dist_all = np.zeros((length, length))
        self.temporal_all = np.zeros((length, length))
        self.temporal_dist_all = np.zeros((length, length))

        self.min_s_m_neq_n = math.inf
        self.min_t_m_neq_n = math.inf
        self.max_s_m_neq_n = -math.inf
        self.max_t_m_neq_n = -math.inf

        self.spatiotemporal_dist_all = np.zeros((length, length))

    def get_max_tw(self):
        return max(np.subtract(self.tws_all[:, 1], self.tws_all[:, 0]))

    def get_min(self, data):
        min_el = math.inf

        for i in range(1, data[0].size):
            for j in range(1, data[0].size):
                if i == j:
                    continue

                if data[i][j] < min_el:
                    min_el = data[i][j]

        return min_el

    def get_max(self, data):
        max_el = -math.inf

        for i in range(1, data[0].size):
            for j in range(1, data[0].size):
                if i == j:
                    continue

                if data[i][j] > max_el:
                    max_el = data[i][j]

        return max_el

    def euclidian_distance(self, i, j):
        if i != j:
            sum_all = 0
            for k in range(len(self.points_all[i])):
                square = pow(self.points_all[j][k] - self.points_all[i][k], 2)
                sum_all += square

            sqr = math.sqrt(sum_all)
            return sqr

        return 0.0

    def fill_euclidian_dist_all(self):
        length = len(self.points_all)
        for i in range(length):
            for j in range(length):
                self.euclidian_dist_all[i, j] = self.euclidian_distance(i, j)

    def Sav1(self, t_cur, i, j):
        return self.k2 * t_cur + self.k1 * self.tws_all[j][1] - (self.k1 + self.k2) * self.tws_all[j][0]

    def Sav2(self, t_cur, i, j):
        return -self.k1 * t_cur + self.k1 * self.tws_all[j][1]

    def Sav3(self, t_cur, i, j):
        return -self.k3 * t_cur + self.k3 * self.tws_all[j][1]

    def D_temporal_integr(self, i, j):
        if i != j:
            customer_i_a_s = self.tws_all[i][0] + self.service_time_all[i] + self.temporal_all[i][j]
            customer_i_b_s = self.tws_all[i][1] + self.service_time_all[i] + self.temporal_all[i][j]

            min_1 = min(customer_i_a_s, self.tws_all[j][0])
            max_1 = min(customer_i_b_s, self.tws_all[j][0])
            integr_1 = quad(self.Sav1, min_1, max_1, args=(i, j))[0]

            min_2 = min(
                max(customer_i_a_s, self.tws_all[j][0]),
                self.tws_all[j][1]
            )
            max_2 = max(
                min(customer_i_b_s, self.tws_all[j][1]),
                self.tws_all[j][0]
            )
            integr_2 = quad(self.Sav2, min_2, max_2, args=(i, j))[0]

            min_3 = max(customer_i_a_s, self.tws_all[j][1])
            max_3 = max(customer_i_b_s, self.tws_all[j][1])
            integr_3 = quad(self.Sav3, min_3, max_3, args=(i, j))[0]

            return self.k1 * self.MAX_TW - (integr_1 + integr_2 + integr_3) / (customer_i_b_s - customer_i_a_s)

        return 0.0

    def fill_temporal_dist_all(self):
        length = self.temporal_dist_all[0].size
        for i in range(length):
            for j in range(length):
                self.temporal_dist_all[i, j] = self.D_temporal_integr(i, j)

    def D_temporal_norm(self, i, j):
        return max(self.temporal_dist_all[i, j],
                   self.temporal_dist_all[j, i])

    def norm_temporal_dist_all(self):
        length = self.temporal_dist_all[0].size
        for i in range(length):
            for j in range(length):
                self.temporal_dist_all[i, j] = self.D_temporal_norm(i, j)

    def D_spatiotemporal(self, i, j):
        if i != j:
            return self.alpha1 * (
                    self.euclidian_dist_all[i][j] - self.min_s_m_neq_n
            ) / (self.max_s_m_neq_n - self.min_s_m_neq_n
                 ) + self.alpha2 * (
                           self.temporal_dist_all[i][j] - self.min_t_m_neq_n) / (
                           self.max_t_m_neq_n - self.min_t_m_neq_n)

        return 0.0

    def fill_spatiotemporal_dist_all(self):
        length = self.spatiotemporal_dist_all[0].size
        for i in range(length):
            for j in range(length):
                self.spatiotemporal_dist_all[i, j] = self.D_spatiotemporal(i, j)

    def calculate_all_distances(self):

        # fill all euclidian distances between points
        self.fill_euclidian_dist_all()

        # init numerically the same temporal distance as spatial
        self.temporal_all = np.copy(self.euclidian_dist_all)

        # calculate all temporal distances and normalize it
        self.fill_temporal_dist_all()
        self.norm_temporal_dist_all()

        # find necessary values for calculating spatiotemporal distances
        self.min_s_m_neq_n = self.get_min(self.euclidian_dist_all)
        self.min_t_m_neq_n = self.get_min(self.temporal_dist_all)
        self.max_s_m_neq_n = self.get_max(self.euclidian_dist_all)
        self.max_t_m_neq_n = self.get_max(self.temporal_dist_all)

        # calculate spatiotemporal distances between all points
        self.fill_spatiotemporal_dist_all()

        return self.spatiotemporal_dist_all
