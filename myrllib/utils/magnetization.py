from scipy import integrate
import math


class Cuboid(object):
    def __init__(self, a, b, h):
        self.a, self.b, self.h = a, b, h
        self.eps = 1e-7

    def Gamma(self, gamma_1, gamma_2, gamma_3):
        l_vec_h = math.sqrt(math.pow(gamma_1, 2) + math.pow(gamma_2, 2) + math.pow(gamma_3 - self.h, 2))
        l_vec_0 = math.sqrt(math.pow(gamma_1, 2) + math.pow(gamma_2, 2) + math.pow(gamma_3, 2))

        Gamma_h = math.log((l_vec_h - gamma_2) / (l_vec_h + gamma_2 + self.eps) + self.eps)
        Gamma_0 = math.log((l_vec_0 - gamma_2) / (l_vec_0 + gamma_2 + self.eps) + self.eps)

        res = Gamma_h - Gamma_0
        return res

    def Psi(self, psi_1, psi_2, psi_3):
        l_vec_h = math.sqrt(math.pow(psi_1, 2) + math.pow(psi_2, 2) + math.pow(psi_3 - self.h, 2))
        l_vec_0 = math.sqrt(math.pow(psi_1, 2) + math.pow(psi_2, 2) + math.pow(psi_3, 2))

        Psi_h = math.atan((psi_1 * (psi_3 - self.h)) / (psi_2 * l_vec_h + self.eps))
        Psi_0 = math.atan((psi_1 * psi_3) / (psi_2 * l_vec_0 + self.eps))

        res = Psi_h - Psi_0
        return res

    def mfi(self, x, y, z):
        if 0 < x < self.a and 0 < y < self.b and 0 < z < self.h:
            x, y, z = self.a, self.b, self.h

        H_x = (-0.5) * (+ self.Gamma(self.a - x, self.b - y, z)
                        + self.Gamma(self.a - x, y, z)
                        - self.Gamma(x, self.b - y, z)
                        - self.Gamma(x, y, z))

        H_y = (-0.5) * (+ self.Gamma(self.b - y, self.a - x, z)
                        + self.Gamma(self.b - y, x, z)
                        - self.Gamma(y, self.a - x, z)
                        - self.Gamma(y, x, z))

        H_z = (-1) * (+ self.Psi(self.b - y, self.a - x, z)
                      + self.Psi(y, self.a - x, z)
                      + self.Psi(self.a - x, self.b - y, z)
                      + self.Psi(x, self.b - y, z)
                      + self.Psi(self.b - y, x, z)
                      + self.Psi(y, x, z)
                      + self.Psi(self.a - x, y, z)
                      + self.Psi(x, y, z))

        H = math.sqrt(math.pow(H_x, 2) + math.pow(H_y, 2) + math.pow(H_z, 2))
        return H


class Sphere(object):
    def __init__(self, a):
        self.a = a
        self.eps = 1e-7
        self.r, self.theta, self.psi = 0, 0, 0

    def ax_transfer(self, x, y, z):
        self.r = math.sqrt(math.pow(x, 2) + math.pow(y, 2) + math.pow(z, 2))
        self.theta = math.acos(z / (self.r + self.eps))
        self.psi = math.atan(y / (x + self.eps))

    def common_term(self, psi_0, theta_0):
        up_term = math.pow(self.a, 3) * math.pow(math.sin(theta_0), 3)
        down_term = math.pow(self.r, 2) + math.pow(self.a, 2) - 2 * self.a * self.r * (
            math.cos(self.theta) * math.cos(theta_0) +
            math.sin(self.theta) * math.sin(theta_0) * math.cos(self.psi - psi_0))
        down_term = math.pow(down_term, 1.5)

        com = up_term / (down_term + self.eps)
        return com

    def fx(self, psi_0, theta_0):
        com = self.common_term(psi_0, theta_0)
        x = math.cos(psi_0) * (self.r * math.cos(self.theta) - self.a * math.cos(theta_0))
        res = com * x
        return res

    def fy(self, psi_0, theta_0):
        com = self.common_term(psi_0, theta_0)
        y = math.sin(psi_0) * (self.r * math.cos(self.theta) - self.a * math.cos(theta_0))
        res = com * y
        return res

    def fz(self, psi_0, theta_0):
        com = self.common_term(psi_0, theta_0)
        z = self.a * math.sin(theta_0) - self.r * math.cos(self.theta) * math.cos(self.psi - psi_0)
        res = com * z
        return res

    def mfi(self, x, y, z):
        self.ax_transfer(x, y, z)
        if self.r < self.a: self.r, self.theta, self.psi = self.a, 0, 0

        H_x = integrate.nquad(self.fx, [[0, 2 * math.pi], [0, math.pi]])
        H_y = integrate.nquad(self.fy, [[0, 2 * math.pi], [0, math.pi]])
        H_z = integrate.nquad(self.fz, [[0, 2 * math.pi], [0, math.pi]])

        H = math.sqrt(math.pow(H_x[0], 2) + math.pow(H_y[0], 2) + math.pow(H_z[0], 2))
        return H


class Sphere_simple(object):
    """
        A simplified version of the calculation for sphere magnet.
        It avoids the complex and time-consuming quadratic integration above.
    """
    def __init__(self, a):
        self.a = a
        self.eps = 1e-7

    def mfi(self, x, y, z):
        r = math.sqrt(math.pow(x, 2) + math.pow(y, 2) + math.pow(z, 2))
        if r < self.a: r = self.a

        H = (2 * math.pow(self.a, 3)) / (3 * math.pow(r, 3) + self.eps)
        # H = 1 / (math.pow(r, 3) + self.eps)

        return H