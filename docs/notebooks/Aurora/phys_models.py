
import numpy as np
from numpy import pi, sqrt, exp, log, log10
import scipy as sp
import scipy.stats



def fn_xyz_pow(x, y, z, theta):
    b, s, k = theta[0], theta[1], theta[2]
    q = sqrt( (x**2 + y**2)/(b*s)**2 + z**2/b**2 )
    f = 1.0 / (1.0 + q)**k
    return f



class Rotation:
    """
    """

    def __init__(self, theta, eta, phi):
        self.theta, self.eta, self.phi = theta, eta, phi

        # Plane rotation matrix
        P = np.array([[ np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi)), 0],
                      [-np.sin(np.deg2rad(phi)), np.cos(np.deg2rad(phi)), 0],
                      [0, 0, 1]])

        # Rodrigues rotation formula
        # https://stackoverflow.com/a/76703318/1038377
        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        zetax = np.sqrt(1-eta**2) * np.cos(np.deg2rad(theta))
        zetay = np.sqrt(1-eta**2) * np.sin(np.deg2rad(theta))
        K = np.array([[0, 0, zetax], [0, 0, zetay], [-zetax, -zetay, 0]])
        R = np.eye(3) + np.sqrt(1-eta**2) * K + (1+eta) * K.dot(K)
        self.T = R.dot(P)



class Density_LBM23_ELZ_DBPLD(Rotation):
    """Density profile for GC/E [M_sun kpc-3]

    Lane, Bovy & Mackereth. The stellar mass of the Gaia-Sausage/Enceladus accretion remnant (2023)
    https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.1209L

    'e-Lz' selection, 'DBPL+D' model.
    """

    def __init__(self):
        self.alpha_1 = 0.64
        self.alpha_2 = 4.08
        self.alpha_3 = 6.8
        self.r_1 = 17.4
        self.r_2 = 34.17
        self.p = 0.61
        self.q = 0.37
        self.f_disk = 0.55
        self.R_sol = 8.12

        super().__init__(theta=77.7, eta=0.99, phi=98.2)


    def _rho_body(self, x, y, z):
        r_ = np.einsum('ij,j...->i...', self.T, np.array([x, y, z]))
        m = np.sqrt(r_[0]**2 + (r_[1]/self.p)**2 + (r_[2]/self.q)**2)
        rho = np.where(m <= self.r_1, \
                       m**(-self.alpha_1), \
                       np.where(m <= self.r_2,
                                self.r_1**(self.alpha_2-self.alpha_1) * m**(-self.alpha_2),
                                self.r_1**(self.alpha_2-self.alpha_1) * self.r_2**(self.alpha_3-self.alpha_2) * m**(-self.alpha_3)))
        return rho


    def _rho_disk(self, x, y, z):
        R = np.sqrt(x**2 + y**2)
        rho = np.exp(-(R - self.R_sol)/2.2 - np.abs(z)/0.8)
        return rho


    def fn(self, x, y, z, *args, **kwargs):
        rho_body = self._rho_body(x, y, z) / self._rho_body(self.R_sol, 0, 0)
        rho_disk = self._rho_disk(x, y, z) / self._rho_disk(self.R_sol, 0, 0)
        rho = (1 - self.f_disk) * rho_body + self.f_disk * rho_disk
        return rho


    def __call__(self, x, y, z, *args, **kwargs):
        return self.fn(x, y, z)



class Density_LBM23_AD_SC(Rotation):
    """Density profile for GC/E [M_sun kpc-3]

    Lane, Bovy & Mackereth. The stellar mass of the Gaia-Sausage/Enceladus accretion remnant (2023)
    https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.1209L

    'AD' selection, 'SC' model.
    """

    def __init__(self):
        self.alpha = -0.57
        self.r_1 = 8.57
        self.p = 0.54
        self.q = 0.46

        super().__init__(theta=147.0, eta=0.84, phi=99.5)


    def fn(self, x, y, z):
        r_ = np.einsum('ij,jk->ik', self.T, np.array([x, y, z]))
        m = np.sqrt(r_[0]**2 + (r_[1]/self.p)**2 + (r_[2]/self.q)**2)
        rho = m**(-self.alpha) * np.exp(-m/self.r_1)
        return rho


    def __call__(self, x, y, z, *args, **kwargs):
        r_ = np.einsum('ij,j...->i...', self.T, np.array([x, y, z]))
        m = np.sqrt(r_[0]**2 + (r_[1]/self.p)**2 + (r_[2]/self.q)**2)
        rho = m**(-self.alpha) * np.exp(-m/self.r_1)
        return rho



class Density_LBM23_AD_BPL(Rotation):
    """Density profile for GC/E [M_sun kpc-3]

    Lane, Bovy & Mackereth. The stellar mass of the Gaia-Sausage/Enceladus accretion remnant (2023)
    https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.1209L

    'AD' selection, 'BPL' model.
    """

    def __init__(self):
        self.alpha_1 = 1.05
        self.alpha_2 = 3.79
        self.r_1 = 23.59
        self.p = 0.55
        self.q = 0.47

        super().__init__(theta=146.6, eta=0.84, phi=98.5)


    def fn(self, x, y, z):
        r_ = np.einsum('ij,jk->ik', self.T, np.array([x, y, z]))
        m = np.sqrt(r_[0]**2 + (r_[1]/self.p)**2 + (r_[2]/self.q)**2)
        rho = np.where(m <= self.r_1, \
                       m**(-self.alpha_1), \
                       self.r_1**(self.alpha_2-self.alpha_1) * m**(-self.alpha_2))
        return rho


    def __call__(self, x, y, z, *args, **kwargs):
        r_ = np.einsum('ij,j...->i...', self.T, np.array([x, y, z]))
        m = np.sqrt(r_[0]**2 + (r_[1]/self.p)**2 + (r_[2]/self.q)**2)
        rho = np.where(m <= self.r_1, \
                       m**(-self.alpha_1), \
                       self.r_1**(self.alpha_2-self.alpha_1) * m**(-self.alpha_2))
        return rho



class Density_tot_BPL:

    def __init__(self, rho0):
        self.rho0 = rho0


    def fn(self, x, y, z, theta):
        b, s, k_1, q_1, k_2 = theta[0], theta[1], theta[2], theta[3], theta[4]
        q2 = (x**2 + y**2)/(b*s)**2 + z**2/b**2
        tmp = q_1**k_2 / (1.0 + q_1**2)**(k_1/2)
        rho = np.where(q2 <= q_1**2, \
                       1.0 / (1.0 + q2)**(k_1/2), \
                       tmp * q2**(-k_2/2))
        return rho


    def __call__(self, x, y, z, theta):
        rho = self.fn(x, y, z, theta)
        xi = theta[5]
        return (1-xi)*rho + xi*self.rho0



def loss_multinomial(theta, dom, proj, TF, fn_xyz, N_lbD):

    xyz = proj.xyz
    rho = fn_xyz(xyz[0], xyz[1], xyz[2], theta)

    p = np.einsum('ijk,ik->ij', TF, rho)
    p /= p.sum()

    mask = p > 0.0
    res = - np.sum(N_lbD[mask] * np.log(p[mask]))

    return res
