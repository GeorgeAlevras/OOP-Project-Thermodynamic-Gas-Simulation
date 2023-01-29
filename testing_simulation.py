from simulation import Container, Simulation, random_balls, time, pressure, distances, relative_distance, velocities, \
    kinetic_energy, total_momentum
import matplotlib.pyplot as plt
import numpy as np

"""
    This file contains all the necessary testing of the simulation.
    It imports the Simulation and Container classes, in order to initialise the simulation, and imports all the
    necessary global variables that store values needed for calculations, e.g. pressure, etc.
    The simulation simulates a group of hydrogen atoms, confined within about 25 angstroms, therefore, the equivalent
    parameters are instantiated below.
    The simulation is only initialised once at the beginning, and all the testing is done below, producing
    graphs to depict relevant information.
    The names of all the figures that are saved are 'parametrised' by the initial conditions, in order
    to remember later which conditions each figure corresponds to.
"""

# In this package the 'balls' are the size of a hydrogen atom
# The container is 50 times the size of a hydrogen atom; 26.5 angstroms
ball_mass = 1.673557693e-27  # Hydrogen atom mass, in units of kg
gas_temperature = 300  # In units of Kelvin
container_radius = 2.65e-9  # In units of m
number_of_balls = 100
radius_of_balls = 5.3e-11  # Hydrogen atom radius, in units of m
number_of_frames = 100

# Simulation is initialised with parameters from above.
c = Container(container_radius)
balls = random_balls(number_of_balls, radius_of_balls, ball_mass, gas_temperature, container_radius)
s = Simulation(balls, c)
s.run(number_of_frames, 0.00001, container_radius, True)  # Number of frames, pause time between frames, show animation

fam = {'fontname': 'Times New Roman'}
params = {
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'font.size': 8,
    'font.family': 'serif',
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 14,
    'figure.figsize': [8, 8]
}

"""
    Plots 1 to 4 test the conservation Laws: Kinetic energy and momentum.
    Plots 1 and 3 check to see that the total KE and momentum remain constant throughout the experiment.
    Plots 2 and 4 look at the KE and momentum fluctuations from the mean, hoping that they are really small.
"""
# This is the kinetic energy of the gas with time
plt.figure(1)
kinetic_energy_norm = [i / 10e-22 for i in kinetic_energy]
average_kinetic_energy = np.average(kinetic_energy_norm)
plt.plot(time, kinetic_energy_norm, label='Average: ' + str(round(average_kinetic_energy, 2)) + 'x10$^-$$^2$$^2$J')
plt.title("Kinetic Energy of Gas", fontsize=16, **fam)
plt.xlabel("Time [s]")
plt.ylabel("Kinetic Energy of Gas [10$^-$$^2$$^2$J]")
plt.legend()
plt.savefig(
    "kinetic_energy_" + str(container_radius) + "_" + str(number_of_balls) + "_" + str(radius_of_balls) + ".png")

# This is the distribution of the energy fluctuations from the mean kinetic energy
plt.figure(2)
energy_fluctuations = [i - np.average(kinetic_energy) for i in kinetic_energy]
avg_fluctuation = np.average(energy_fluctuations)
n_1, bins_1, patches_1 = plt.hist(x=energy_fluctuations, bins=12, color='#708090', alpha=0.7, rwidth=0.9,
                                  label='Average: ' + str(round(avg_fluctuation, 39)))
plt.title("Distribution of Energy Fluctuations from the Mean KE", fontsize=14, **fam)
plt.xlabel("Energy Fluctuations [J]")
plt.ylabel("Frequency")
plt.legend()
plt.savefig(
    "energy_fluctuations_" + str(container_radius) + "_" + str(number_of_balls) + "_" + str(radius_of_balls) + ".png")

# This is the total momentum of the gas with time
plt.figure(3)
total_momentum_norm = [i / (10e-25) for i in total_momentum]
average_momentum = np.average(total_momentum_norm)
plt.plot(time, total_momentum_norm, label='Average: ' + str(round(average_momentum, 3)) + 'x10$^-$$^2$$^5$kgms$^-$$^1$')
plt.title("Total Momentum of Gas", fontsize=16, **fam)
plt.xlabel("Time [s]")
plt.ylabel("Momentum of Gas [10$^-$$^2$$^2$kgms$^-$$^1$]")
plt.legend()
plt.savefig(
    "total_momentum_" + str(container_radius) + "_" + str(number_of_balls) + "_" + str(radius_of_balls) + ".png")

# This is the distribution of the energy fluctuations from the mean kinetic energy
plt.figure(4)
momentum_fluctuations = [i - np.average(total_momentum) for i in total_momentum]
avg_fluctuation_mom = np.average(momentum_fluctuations)
n_2, bins_2, patches_2 = plt.hist(x=momentum_fluctuations, bins=12, color='#708090', alpha=0.7, rwidth=0.9,
                                  label='Average: ' + str(round(avg_fluctuation_mom, 43)))
plt.title("Distribution of Momentum Fluctuations from the Mean Momentum", fontsize=14, **fam)
plt.xlabel("Momentum Fluctuations [kgms$^-$$^1$]")
plt.ylabel("Frequency")
plt.legend()
plt.savefig(
    "momentum_fluctuations_" + str(container_radius) + "_" + str(number_of_balls) + "_" + str(radius_of_balls) + ".png")

"""
    Plot 5 looks at the pressure of the gas, wanting to ensure that it remains constant throughout,
    as the volume and temperature of the gas is constant, and the number of balls (hydrogen atoms) is also constant.
"""
# This is the pressure of the gas with time
plt.figure(5)
avg_pressure = pressure[len(pressure) - 1]
plt.plot(time, pressure, label="Average Pressure: " + str(round(avg_pressure, 4)) + "Pa")
plt.title("Pressure of Gas", fontsize=16, **fam)
plt.xlabel("Time [s]")
plt.ylabel("Pressure of Gas [Pa]")
plt.legend()
plt.savefig("pressure_" + str(container_radius) + "_" + str(number_of_balls) + "_" + str(radius_of_balls) + ".png")

"""
    Plots 6 and 7 test that no ball escapes the container and that there is no overlap of balls
"""
# This is the distribution of the distances of each ball
# from the centre of the container, taken for each frame
plt.figure(6)
avg_distance = round(np.average(distances), 2)
n_3, bins_3, patches_3 = plt.hist(x=distances, bins=30, color='#27D93C', alpha=0.7, rwidth=0.9)
plt.title("Distribution of Distances of Balls from Centre", fontsize=16, **fam)
plt.xlabel("Distance from container centre [m]")
plt.ylabel("Frequency")
ticks = np.linspace(0, container_radius, 6)
plt.xticks(ticks, ['0', '', '', '', '', 'R-r'])
plt.savefig(
    'distance_from_centre_' + str(container_radius) + '_' + str(number_of_balls) + '_' + str(radius_of_balls) + '.png')

# This is the distribution of the relative distances of all balls
# taken for each frame
plt.figure(7)
avg_relative_distance = round(np.average(relative_distance), 2)
n_4, bins_4, patches_4 = plt.hist(x=relative_distance, bins=30, color='#27D93C', alpha=0.7, rwidth=0.9)
plt.title("Distribution of Relative Distances of Balls", fontsize=16, **fam)
plt.xlabel("Relative distance of balls [m]")
plt.ylabel("Frequency")
ticks_1 = np.linspace(2 * radius_of_balls, 2 * container_radius - 2 * radius_of_balls, 10)
plt.xticks(ticks_1, ['2r', '', '', '', '', '', '', '', '', '2R-2r'])
plt.yticks(size='6')
plt.savefig(
    'relative_distance_' + str(container_radius) + '_' + str(number_of_balls) + '_' + str(radius_of_balls) + '.png')

"""
    Plot 8 looks at the velocity distribution of the balls, and that they follow the Maxwell-Boltzmann Distribution.
"""


def get_boltzmann(vel):  # This function returns the Maxwell-Boltzmann probability density for a given velocity
    return 4 * np.pi * ((ball_mass / (2 * np.pi * 1.38064852e-23 * (gas_temperature / number_of_balls))) ** 1.5) * (
            vel ** 2) * np.exp(
        -(ball_mass * (vel ** 2)) / (2 * 1.38064852e-23 * (gas_temperature / number_of_balls)))


plt.figure(8)
avg_velocity = round(np.average(velocities), 2)
x = np.linspace(0, 1000, 10000)
boltzmann = [4 * np.pi * ((ball_mass / (2 * np.pi * 1.38064852e-23 * (gas_temperature / number_of_balls))) ** 1.5) * (
        i ** 2) * np.exp(-(ball_mass * (i ** 2)) / (2 * 1.38064852e-23 * (gas_temperature / number_of_balls))) for i
             in x]
v_p = np.sqrt((2 * 1.38064852e-23 * (gas_temperature / number_of_balls)) / ball_mass)
v_avg = np.sqrt((8 * 1.38064852e-23 * (gas_temperature / number_of_balls)) / (np.pi * ball_mass))
v_rms = np.sqrt((3 * 1.38064852e-23 * (gas_temperature / number_of_balls)) / ball_mass)
n_5, bins_5, patches_5 = plt.hist(x=velocities, bins=80, color='#27D9FF', alpha=0.7, rwidth=0.9, density=True,
                                  label="Velocity Distribution, \nAverage: " + str(avg_velocity) + "ms$^-$$^1$")
plt.plot(x, boltzmann, color='#FA6C33', label="Maxwell-Boltzmann Curve")
extra_ticks = [v_p, v_avg, v_rms]
plt.xticks(extra_ticks, ['v_p', 'v_μ', 'v_r'], size=6)
x_points_v_p = [v_p, v_p]
y_points_v_p = [0, get_boltzmann(v_p)]
x_points_v_avg = [v_avg, v_avg]
y_points_v_avg = [0, get_boltzmann(v_avg)]
x_points_v_rms = [v_rms, v_rms]
y_points_v_rms = [0, get_boltzmann(v_rms)]
plt.plot(x_points_v_p, y_points_v_p, 'o--')
plt.plot(x_points_v_avg, y_points_v_avg, 'o--')
plt.plot(x_points_v_rms, y_points_v_rms, 'o--')
plt.xlim(-25, 1000)
plt.title("Distribution of Velocities of Balls", fontsize=16, **fam)
plt.xlabel("Velocity of balls [ms$^-$$^1$]")
plt.ylabel("Probability Density")
plt.legend()
plt.savefig('velocity_' + str(container_radius) + '_' + str(number_of_balls) + '_' + str(radius_of_balls) + '.png')

plt.show()
