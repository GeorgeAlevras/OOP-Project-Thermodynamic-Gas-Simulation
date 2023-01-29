from simulation import Container, Simulation, random_balls, pressure
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

"""
    This file checks the pressure laws for a gas with 'balls' the size of a hydrogen atom.
    The simulation is checked for:
        a) Pressure against temperature(Plot 1)
        b) Pressure against the inverse of volume (Plot 2)
        c) Pressure against the number of balls in the container (Plot 3)
"""

# In this package the 'balls' are the size of a hydrogen atom
# The container is 50 times the size of a hydrogen atom; 26.5 angstroms
ball_mass = 1.673557693e-27  # Hydrogen atom mass, in units of kg
container_radius = 2.65e-7  # In units of m
number_of_balls = 100
radius_of_balls = 5.3e-11  # In units of m
number_of_frames = 600

fam = {'fontname': 'Times New Roman'}
params = {
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'font.size': 8,
    'font.family': 'serif',
    'legend.fontsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.figsize': [8, 8]
}

# Initially the temperature is varied, and all other parameters are kept constant
gas_temperature = [50, 100, 150, 200, 250, 300, 350, 400]
pressures_varying_T = []
for x in range(len(gas_temperature)):
    c = Container(container_radius)
    balls = random_balls(number_of_balls, radius_of_balls, ball_mass, gas_temperature[x], container_radius)
    s = Simulation(balls, c)
    s.run(number_of_frames, 0.00001, container_radius, False)
    pressures_varying_T.append((pressure[len(pressure) - 1]))

plt.figure(1)
x = np.linspace(50, 400, 10000)
y = [6.5349e-2 * i for i in x]
pressures_t_mili = [(i * 1000) for i in pressures_varying_T]
plt.plot(gas_temperature, pressures_t_mili, 'x', color='r')
fit_phase, cov_phase = sp.polyfit(gas_temperature, pressures_t_mili, 1, cov=True)
p_phase = sp.poly1d(fit_phase)
plt.plot(gas_temperature, p_phase(gas_temperature), color='k',
         label='Linear fit, gradient = ' + str(round(fit_phase[0], 7)) + "+-" + str(round(cov_phase[0][0], 6)) +
         "\n b = 1.113x10$^-$$^2$$^0$ +- 0.001x10$^-$$^2$$^0$")
plt.plot(x, y, color='b', label='Theoretical Van Der Waals')
plt.title("Pressure of Gas Varying Temperature", fontsize=16, **fam)
plt.xlabel("Temperature [K]")
plt.ylabel("Pressure of Gas [mPa]")
plt.legend()
print('Gradient is: ', (number_of_balls * 1.38064852e-23) / (np.pi * container_radius ** 2))
plt.savefig('pressure_against_T_H1.png')

# The gas is given a single temperature again, and the container radius is varied
gas_temperature = 300
container_radius = [2.2e-9, 2.4e-9, 2.6e-9, 2.8e-9, 3e-9]
pressures_varying_V = []
for x in range(len(container_radius)):
    c = Container(container_radius[x])
    balls = random_balls(number_of_balls, radius_of_balls, ball_mass, gas_temperature, container_radius[x])
    s = Simulation(balls, c)
    s.run(number_of_frames, 0.00001, container_radius, False)
    pressures_varying_V.append(np.average(pressure))

plt.figure(2)
container_area_inverse = [1 / (2 * np.pi * (i ** 2)) for i in container_radius]
pressures_v_mili = [(i * 1000) for i in pressures_varying_V]
plt.plot(container_area_inverse, pressures_v_mili, 'x', color='r')
fit_phase_1, cov_phase_1 = sp.polyfit(container_area_inverse, pressures_v_mili, 1, cov=True)
p_phase_1 = sp.poly1d(fit_phase_1)
plt.plot(container_area_inverse, p_phase_1(container_area_inverse), color='k',
         label='gradient = ' + str(round(fit_phase_1[0], 6)) + "+-" + str(round(cov_phase_1[0][0], 11)))
plt.title("Pressure of Gas Varying V", fontsize=16, **fam)
plt.xlabel("Inverse of Area [m$^-$$^2$]")
plt.ylabel("Pressure of Gas [mPa]")
plt.legend()
print('Gradient is: ', (number_of_balls * 1.38064852e-23 * gas_temperature))
plt.savefig('pressure_against_V_H1.png')

# The container radius is brought back to the original, and the number of balls is varied
container_radius = 2.65e-9
number_of_balls = [40, 60, 80, 100, 120]
pressures_varying_N = []
for x in range(len(number_of_balls)):
    c = Container(container_radius)
    balls = random_balls(number_of_balls[x], radius_of_balls, ball_mass, gas_temperature, container_radius)
    s = Simulation(balls, c)
    s.run(number_of_frames, 0.00001, container_radius, False)
    pressures_varying_N.append(np.average(pressure))

plt.figure(3)
pressures_n_mili = [(i * 1000) for i in pressures_varying_N]
plt.plot(number_of_balls, pressures_n_mili, 'x', color='r')
fit_phase_2, cov_phase_2 = sp.polyfit(number_of_balls, pressures_n_mili, 1, cov=True)
p_phase_2 = sp.poly1d(fit_phase_2)
plt.plot(number_of_balls, p_phase_2(number_of_balls), color='k',
         label='gradient = ' + str(round(fit_phase_2[0], 6)) + "+-" + str(round(cov_phase_2[0][0], 11)))
plt.title("Pressure of Gas Varying N", fontsize=16, **fam)
plt.xlabel("Number of balls, N")
plt.ylabel("Pressure of Gas [mPa]")
plt.legend()
print('Gradient is: ', (1.38064852e-23 * gas_temperature) / (np.pi * container_radius ** 2))
plt.savefig('pressure_against_N_H1.png')

plt.show()
