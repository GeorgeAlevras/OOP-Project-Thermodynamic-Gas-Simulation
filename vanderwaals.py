from simulation import Container, Simulation, random_balls, pressure
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# In this package the 'balls' are the size of a hydrogen atom
# The container is 50 times the size of a hydrogen atom; 26.5 angstroms

ball_mass = 1.673557693e-27  # In units of kg
container_radius = 2.65e-8  # In units of m
number_of_balls = 100
radius_of_balls = 5.3e-11  # In units of m
number_of_frames = 1000

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

gas_temperatures = [200, 225, 250, 275, 300, 325, 350, 375, 400]
pressures_varying_T = []
for x in range(len(gas_temperatures)):
    print((x/len(gas_temperatures)))
    c = Container(container_radius)
    balls = random_balls(number_of_balls, radius_of_balls, ball_mass, gas_temperatures[x], container_radius)
    s = Simulation(balls, c)
    s.run(number_of_frames, 0.00001, container_radius, False)
    pressures_varying_T.append(np.average(pressure))


plt.figure(1)
plt.plot(gas_temperatures, pressures_varying_T, 'x')
fit_phase, cov_phase = sp.polyfit(gas_temperatures, pressures_varying_T, 1, cov=True)
p_phase = sp.poly1d(fit_phase)
volume = (np.pi*container_radius**2)
b = (volume - (number_of_balls*1.38064852e-23)/fit_phase[0])/number_of_balls
db = (1.38064852e-23/(fit_phase[0]**2) * cov_phase[0][0])
plt.plot(gas_temperatures, p_phase(gas_temperatures), color='k', label='b = ' + str(b) + "+-" + str(db))
plt.title("Pressure of Gas Varying T", fontsize=16, **fam)
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure of Gas (Pa)")
plt.legend()
plt.savefig('van_der_waals.png')
