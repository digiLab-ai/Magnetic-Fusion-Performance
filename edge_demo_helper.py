
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
ELECTRIC_CHARGE = 1.602176634e-19  # C
EFFECTIVE_CHARGE = 1.
ION_MASS = 2. * 1.67e-27  # (kg)

cmap = "summer"
uncert_cmap = "autumn_r"

# Conversions
def convert_eV_to_K(temperature):
    return temperature * ELECTRIC_CHARGE / BOLTZMANN_CONSTANT

def convert_eV_to_J(temperature):
    return temperature * ELECTRIC_CHARGE

def convert_K_to_eV(temperature):
    return temperature * BOLTZMANN_CONSTANT / ELECTRIC_CHARGE

# Helper functions

def random_log_space(lwr, upr, size):
    return np.exp(np.random.uniform(*np.log([lwr, upr]), size))

def log_space(lwr, upr, num):
    return np.exp(np.linspace(*np.log([lwr, upr]), num))

class SputteringModel:

    def __init__(self,
                 S=0.042, 
                 p=0.2,
                 q=1.,
                 E_th=1.e-4  # eV
                 ):
        self.S = S  # The material relevant constant
        self.p = p 
        self.q = q
        self.E_th = E_th  # The sputtering threshold energy

    def __call__(self, te_t, ne_t):
        return ne_t, te_t, self.ion_flux(te_t=te_t, ne_t=ne_t), self.sputtering_rate(te_t=te_t, ne_t=ne_t), self.sputtering_yield(te_t=te_t)
    
    def ion_flux(self, ne_t, te_t):
        return ne_t * self.ion_speed(te_t)
    
    def ion_speed(self, te_t):
        return np.sqrt(2 * BOLTZMANN_CONSTANT * convert_eV_to_K(te_t) / ION_MASS)
    
    def sputtering_rate(self, ne_t, te_t):
        return self.sputtering_yield(te_t = te_t) * self.ion_flux(ne_t=ne_t, te_t=te_t)
    
    def sputtering_yield(self, te_t):
        e_i = self.calc_ion_energy(te_t)
        return self.S * (max(e_i - self.E_th, 0.) **self.p) / (self.E_th **self.q)

    def calc_ion_energy(self, te_t):
        return te_t
    

class TwoPointModel:
    gamma = 7.  # sheath coefficient
    kappa = 2000.
    gamma_flow = 1.  # target electron flow coefficient: the 1 is isothermal, 5/3 is adiabatic
    E = 30. * 1.6e-19
    
    def __init__(self,
                 upstream_q=100.,
                 upstream_density=1.e19,
                 parallel_connection_length=5000.,
                 # Modified 2 point model relevant
                 f_c=1.,
                 f_mom=1.,
                 f_power=0.,
                 # Sputtering model relevant
                 **sputtering_kwargs
                 ):
        upstream_q = upstream_q
        upstream_density = upstream_density
        parallel_connection_length = parallel_connection_length
        
        f_power = self.calc_f_pow(upstream_density)

        self.q_u = upstream_q
        self.kappa = convert_eV_to_K(self.kappa)
        self.n_u = upstream_density
        self.L = parallel_connection_length
        self.f_c = max(0.0001, min(f_c, 1.))
        self.f_m = max(0.0001, min(f_mom, 1.))
        self.f_p = max(0., min(f_power, 0.9999))

        self.coef = (7. * self.L * self.f_c * self.q_u) / (2. * self.kappa)        
        alpha = np.sqrt(self.gamma_flow * EFFECTIVE_CHARGE * BOLTZMANN_CONSTANT / ION_MASS) 
        self.const =  - ( (2. * self.q_u * (1. - self.f_p)) / (self.f_m * self.gamma * alpha * self.n_u)) ** (7/2)
        self.sputtering_model = SputteringModel(**sputtering_kwargs)
 
    def calc_e_loss(self, t_t):
        f_m = self.calc_f_mom(t_t)
        t_u = self.get_upstream_electron_temperature(t_t)
        n_t = (f_m * self.n_u * t_u) / (2. * t_t)
        c_s = self.calc_ion_sound_speed(t_t)
        
        e =  n_t * c_s * self.E

        # print(f"fm, {f_m}")
        # print(f"nt, {n_t}")
        # print(f"tt, {t_t}")
        # print(f"c_s, {c_s}")
        # print(f"e, {e}")

        return e * 0.
    
    def calc_ion_sound_speed(self, t_t):
        return np.sqrt(convert_eV_to_J(t_t)/ ION_MASS)

    def calc_f_pow(self, n, a=1.e19, b=1./1.e19):
        return 1 / (1 + np.exp(-b * (n - a)))
    
    def calc_f_mom(self, t_t):
        return 1.
        # return 0.9 * ((1. - np.exp( - t_t / 2.1)) ** 2.9)
        # return 0.9 * ((1. - np.exp( - t_t / 0.01)) ** 2.9)

    def __call__(self):
        initial_guess = ((7 * self.L *self.f_c * self.q_u / (2. * self.kappa)) + (0.01 ** (7/2)) ) ** (2/7)
        
        initial_guess = 1.e-8
        
        # initial_guess = 1.e-2
        target_temperature = self.find_target_temperature(initial_guess)
        
        # target_temperature = max(target_temperature, 1.)
        target_electron_density = self.get_target_electron_density(target_temperature=target_temperature)
        # print(self.get_upstream_electron_temperature(target_temperature))
        # print(f"t_t: {target_temperature:2e}")
        # print(f"q_t: {self.get_target_q(target_temperature, target_electron_density):2e}")
        nt, tt, ion_flux, sputtering, sputtering_yield = self.sputtering_model(te_t=target_temperature, ne_t=target_electron_density)

        return ion_flux
    
    def find_target_temperature(self, initial_guess):
        target_temperature = fsolve(self.root_function, initial_guess)
        target_temperature = target_temperature[0]
        diff = abs(target_temperature - initial_guess) / initial_guess
        if diff < 5.e-1 or target_temperature < 1.e-5:
            return self.find_target_temperature(initial_guess * 10.)
        else:
            return target_temperature
    
    def get_target_electron_density(self, target_temperature):
        t_u = self.get_upstream_electron_temperature(target_temperature=target_temperature)
        return (self.f_m * self.n_u * t_u) / (2. * target_temperature)
    
    def get_upstream_electron_temperature(self, target_temperature):
        return (target_temperature ** (7/2) + 7 * self.L * self.f_c * self.q_u / (2. * self.kappa)) ** (2/7)
    
    def get_target_q(self, t_t, n_t):
        c_s = np.sqrt(BOLTZMANN_CONSTANT * convert_eV_to_K(t_t) / (ION_MASS))
        return (n_t * c_s * self.gamma * t_t * ELECTRIC_CHARGE)

    def root_function(self,
                    target_temperature
                    ):
        t_t = target_temperature  # convert_eV_to_K(target_temperature)
        f_m = self.calc_f_mom(target_temperature)
        a = (f_m * self.n_u * t_t )** (7/2)
        b = (7/2) * ((f_m * self.n_u) ** (7/2)) * self.L * self.f_c * self.q_u / self.kappa
        
        
        e = self.calc_e_loss(t_t) 

        c = ((np.sqrt(ION_MASS / (ELECTRIC_CHARGE * t_t)) * 2 * self.q_u * (1. - self.f_p)/ (ELECTRIC_CHARGE * self.gamma)) - (2. * e * (1. - self.f_p)) / (ELECTRIC_CHARGE* self.calc_ion_sound_speed(t_t) *self.gamma)) ** (7/2)
        return (a + b - c)
    
def ion_flux_model(*args, **kwargs):
    return TwoPointModel(*args, **kwargs)()




def plot_one_d(
        x_quantity_name="",
        y_quantity_name="",
        y_quantity_std_name="",
        train_df=None,
        prediction_df=None,
        x_train=None,
        y_train=None,
        x_prediction=None,
        y_prediction=None,
        y_std_prediction=None,
):
    
    fig, ax = plt.subplots()

    if train_df is not None:
        ax.scatter(train_df[x_quantity_name],
                   train_df[y_quantity_name],
                      marker='o', color='black')

    if prediction_df is not None:
        ax.plot(prediction_df[x_quantity_name],
                prediction_df[y_quantity_name],
                   linestyle='dashed', color='black')
        ax.fill_between(
                        prediction_df[x_quantity_name],
                        prediction_df[y_quantity_name]-prediction_df[y_quantity_std_name],
                        prediction_df[y_quantity_name]+prediction_df[y_quantity_std_name],
                        alpha=0.5, color='red')

    ax.set_ylabel(y_quantity_name)
    ax.set_xlabel(x_quantity_name)

    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()