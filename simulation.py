import numpy as np
import pylab as pl
import random as rn
import scipy.stats as stats

"""
    This is the file containing all the 'simulation' classes and methods.
    This file will be imported by the testing and other scripts to 'run' the simulation and test
    all the physics of it.
    All methods are completely parametrised, avoiding any hard-coding, making them very flexible.
    Each individual 'task' and 'sub-task' has been made into a method, making the code modular and more concise.#
    Use of recursion has been used instead of loops to optimise the number of calculations and 
    increase the time efficiency of the algorithms.
"""

# Global variables used for testing and calculations
momentum_change = 0
total_time = 0
time = []
pressure = []
distances = []
relative_distance = []
kinetic_energy = []
velocities = []
total_momentum = []


class Ball:
    """
    This class defines the ball as an object
    A ball is initialised with: mass, radius, position and velocity.
    It also has a circle patch attribute, in order to be displayed.
    The key methods are the time_to_collision and collide methods, the first of which calculates the time
    to collision for two ball objects and the second of which collides them (changing their velocities) respectively.
    """

    def __init__(self, mass, radius, position, vel):
        self._m = mass
        self._r = radius
        self._p = np.array(position, dtype=float)
        self._v = np.array(vel, dtype=float)
        self._patch = pl.Circle(self._p, self._r, fc='r', ec='g')

    def __repr__(self):
        pos = [self._p[0], self._p[1]]
        vel = [self._v[0], self._v[1]]
        return "%s %r\n%s %r\n%s %r\n%s %r" % (
            "Mass: ", self._m, "Radius: ", self._r, "Position: ", pos, "Velocity: ", vel)

    def __str__(self):
        pos = [self._p[0], self._p[1]]
        vel = [self._v[0], self._v[1]]
        return "%r\n%r\n%r\n%r" % (self._m, self._r, pos, vel)

    def get_pos(self):
        return self._p

    def get_vel(self):
        return self._v

    def get_patch(self):
        return self._patch

    def get_distance(self):  # Returns the distance between a ball object and the container centre
        return np.sqrt(self._p[0] ** 2 + self._p[1] ** 2)

    def get_relative_distance(self, other):  # Returns the distance between two ball objects
        relative = self._p - other._p
        return np.sqrt(relative[0] ** 2 + relative[1] ** 2)

    def get_speed(self):
        speed = np.sqrt(self._v[0] ** 2 + self._v[1] ** 2)
        return speed

    def get_kinetic_energy(self):
        v_sq = (self.get_speed()) ** 2
        kin_energy = 0.5 * v_sq * self._m
        return kin_energy

    def get_momentum(self):
        return self._m * self.get_vel()

    def pressure(self, radius):
        force = 50 * momentum_change / total_time
        press = force / (2 * np.pi * radius)  # The 'area' of the container is the container's circumference
        return press

    def move(self, dt):
        # The -ve increment ensures the ball never quite makes it to perfectly touching
        # with another ball (or container), to avoid 0 time to collision, which would cause errors
        p_new = self._p + (dt - 0.0000000000000001) * self._v
        self._p = p_new
        self._patch.center = self._p

    def time_to_collision(self, other):
        # Ball and container are differentiated form their class
        # This makes the method flexible - it can be used both for a ball or container object
        is_either_container = isinstance(self, Container) or isinstance(other, Container)

        if is_either_container is False:
            r = self._p - other._p
            v = self._v - other._v
            R = self._r + other._r  # Adding, because they're two balls (no overlap)
        else:
            r = self._p - other._p
            v = self._v - other._v
            R = self._r - other._r  # Subtracting, because it's a ball and a container (ball is within container)

        # Terms as they appear in quadratic equation for dt
        a = np.dot(v, v)
        b = 2 * (np.dot(v, r))
        c = np.dot(r, r) - R * R

        if (b ** 2) < (4 * a * c):  # Discriminant will determine equation; if negative => no real solutions
            return None
        else:  # Otherwise, keep the smallest time, i.e. the first collision that will occur
            dt_1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            dt_2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

            if (dt_1 > 0) and (dt_2 > 0):
                l = [dt_1, dt_2]
                return min(l)
            elif (dt_1 > 0) and (dt_2 < 0):
                return dt_1
            elif (dt_2 > 0) and (dt_1 < 0):
                return dt_2
            else:
                return None

    def collide(self, other):  # Collides two objects (ball and ball, or ball and container)
        dot_1 = np.dot(self._v - other._v, self._p - other._p)
        dot_2 = np.dot(other._v - self._v, other._p - self._p)
        x_1 = self._p - other._p
        x_2 = other._p - self._p
        v1_new = self._v - (((2 * other._m) / (self._m + other._m)) * (dot_1 / (x_1[0] ** 2 + x_1[1] ** 2)) * x_1)
        v2_new = other._v - (((2 * self._m) / (self._m + other._m)) * (dot_2 / (x_2[0] ** 2 + x_2[1] ** 2)) * x_2)

        self._v = v1_new
        other._v = v2_new

        # Calculates the momentum change of the container, if a ball collided with it
        if isinstance(other, Container):
            global momentum_change
            momentum_change += 2 * self._m * self.get_speed()


class Container(Ball):
    """
    This class defines the container as an object
    A container inherits from a ball (a subclass)
    A container is initialised with fewer variables:
        radius
    And also has a circle patch attribute, to be displayed
    A container is a specific and unique type of ball, so
    it is parametrised with specific values:
        a) a large mass of 10^20 kg, to approximate an infinitely massive container
        c) [0, 0] position and velocity vectors
    The container inherits all the methods of the Ball class
    """

    def __init__(self, radius):
        super().__init__(mass=10 ** 20, radius=radius, position=np.array([0, 0]), vel=np.array([0, 0]))
        self._patch = pl.Circle(self._p, self._r, ec='b', fill=False)


class Simulation:
    """
    This class defines the Simulation as an object
    A simulation object is initialised with:
        balls (a list of ball objects), container (a container object)
    The key methods include:
    load_all_times: which returns the time to collision of all balls with
        all other balls, and all balls with the container
    next_collision: which finds the next minimum time to collision, moves all
        the balls in that time, and collides the ball(s) with the shortest time to collision
    run: runs the simulation, according to some parameters
    """

    def __init__(self, balls, container):
        self._balls = balls
        self._container = container

    # This function returns two lists, containing the times it will take
    # for each ball to collide with each other ball, and the container
    def load_all_times(self):
        global relative_distance
        dt_balls = []
        dt_container = []
        counter = 0  # The counter ensures no time of collision for two balls is calculated more than once
        for x in range(len(self._balls)):
            if self._balls[x].time_to_collision(self._container) is not None:
                dt_container.append([x, self._balls[x].time_to_collision(self._container)])
            for i in range(counter, len(self._balls)):
                if i != x:  # No time of collision for a ball with itself is calculated
                    relative_distance.append(self._balls[x].get_relative_distance(self._balls[i]))
                    if self._balls[x].time_to_collision(self._balls[i]) is not None:
                        dt_balls.append([x, i, self._balls[x].time_to_collision(self._balls[i])])
            counter += 1
        return dt_balls, dt_container

    # This method returns the ball that has the shortest collision time
    # with the container, and the time it will take to collide
    def container_minimums(self):
        dt_balls, dt_container = self.load_all_times()
        time_c = []
        time_c_index = []
        for x in range(len(dt_container)):
            time_c.append(dt_container[x][1])
            time_c_index.append(dt_container[x][0])
        dt_c_min = min(time_c)
        dt_c_index_min = dt_container[time_c.index(dt_c_min)][0]
        return dt_c_min, dt_c_index_min

    # This method returns the two balls with the smallest collision time
    # and the time it will take for them to collide
    def balls_minimums(self):
        dt_balls, dt_container = self.load_all_times()
        time_b = []
        index_b_i = []
        index_b_j = []
        for x in range(len(dt_balls)):
            time_b.append(dt_balls[x][2])
            index_b_i.append(dt_balls[x][0])
            index_b_j.append(dt_balls[x][1])
        if len(time_b) == 0:
            dt_b_min = 0
            dt_b_index_min_1 = 0
            dt_b_index_min_2 = 0
        else:
            dt_b_min = min(time_b)
            index = time_b.index(dt_b_min)
            dt_b_index_min_1 = index_b_i[index]
            dt_b_index_min_2 = index_b_j[index]
        return dt_b_min, dt_b_index_min_1, dt_b_index_min_2

    # Minimum collision time is found, balls are moved to new position and collided (their velocities changed)
    def next_collision(self):
        global distances
        global kinetic_energy
        global velocities
        global time
        global total_time
        global total_momentum
        kinetic_energy_each_ball = []
        momentum_each_ball = []

        #  The minimum times (and their corresponding ball/container objects) are loaded
        dt_c_min, dt_c_index_min = self.container_minimums()
        dt_b_min, dt_b_index_min_1, dt_b_index_min_2 = self.balls_minimums()

        # This is the case where the minimum time certainly comes from a ball-container collision
        # in the rare case that no ball crosses paths with any other ball, hence the list is empty
        if dt_b_min == 0:
            d_min = dt_c_min
            total_time += d_min
            time.append(total_time)
            for x in range(len(self._balls)):
                # Kinetic energy, momentum and distances for each ball are calculated
                kinetic_energy_each_ball.append(self._balls[x].get_kinetic_energy())
                momentum_each_ball.append(self._balls[x].get_momentum())
                distances.append(self._balls[x].get_distance())
                velocities.append(self._balls[x].get_speed())
                self._balls[x].move(d_min)  # All balls are moved in that time
            self._container.move(dt_c_min)  # Ball is collided with the container
            self._balls[dt_c_index_min].collide(self._container)
            kinetic_energy.append(np.sum(kinetic_energy_each_ball))
            momentum_each_ball.append(self._container.get_momentum())
            total_momentum.append(np.linalg.norm(np.add(momentum_each_ball)))
        else:
            d_min_array = [dt_c_min, dt_b_min]
            d_min = min(d_min_array)
            total_time += d_min
            time.append(total_time)
            for x in range(len(self._balls)):
                # Kinetic energy, momentum and distances for each ball are calculated
                kinetic_energy_each_ball.append(self._balls[x].get_kinetic_energy())
                momentum_each_ball.append(self._balls[x].get_momentum())
                distances.append(self._balls[x].get_distance())
                velocities.append(self._balls[x].get_speed())
                self._balls[x].move(d_min)  # All balls are moved in that time
            kinetic_energy.append(np.sum(kinetic_energy_each_ball))
            momentum_each_ball.append(self._container.get_momentum())
            total_momentum.append(np.linalg.norm(np.sum(momentum_each_ball)))
            if dt_c_min < dt_b_min:  # This is for a ball-container collision
                self._container.move(dt_c_min)
                self._balls[dt_c_index_min].collide(self._container)
            else:  # This is for a ball-ball collision
                self._balls[dt_b_index_min_1].collide(self._balls[dt_b_index_min_2])

    # This method runs the simulation and presents the data visualisation of the hard-sphere gas
    def run(self, num_frames, pause_time, axes_limits, animate=False):
        if animate:
            fam = {'fontname': 'Times New Roman'}
            params = {
                'axes.labelsize': 12,
                'axes.titlesize': 16,
                'font.size': 8,
                'font.family': 'serif',
                'legend.fontsize': 12,
                'xtick.labelsize': 14,
                'ytick.labelsize': 14,
                'figure.figsize': [8, 8]
            }
            pl.rcParams.update(params)
            # The axes of the graph are parametrised by the container radius
            ax = pl.axes(xlim=(-(axes_limits + 0.02 * axes_limits), (axes_limits + 0.02 * axes_limits)),
                         ylim=(-(axes_limits + 0.02 * axes_limits), (axes_limits + 0.02 * axes_limits)))
            pl.title("Thermodynamic Simulation", fontsize=16, **fam)
            pl.xlabel("Horizontal distance from container centre (m)")
            pl.ylabel("Vertical distance from container centre (m)")
            ax.add_artist(self._container.get_patch())
            for x in range(len(self._balls)):
                ax.add_patch(self._balls[x].get_patch())
        for frame in range(num_frames):
            print(str(round((frame / num_frames) * 100, 2)) + "%")
            self.next_collision()
            if animate:
                pl.pause(pause_time)
            pressure.append(self._container.pressure(self._container._r))
        if animate:
            pl.show()


# This method returns a list of velocities (for all balls), so that their total temperature
# corresponds to the desired gas temperature (therefore the gas temperature can be controlled)
# It is parametrised by the number of balls and their mass to make it flexible, avoiding hard-coding
def velocities_at_given_temp(num_balls, mass, total_temperature):
    ball_temperatures = []
    max_temperature = total_temperature
    for x in range(num_balls - 1):
        # The temperature of the next ball is bounded by the maximum (gas temperature)
        # The temperature of the next ball is picked randomly with a Gaussian distribution
        # centred around (1/number_of_balls)*gas_temperature
        low, high = -1, (max_temperature - (1 / num_balls) * total_temperature) / ((1 / num_balls) * total_temperature)
        temperature = stats.truncnorm(low, high, loc=(1 / num_balls) * total_temperature,
                                      scale=(1 / num_balls) * total_temperature)
        ball_temperatures.append(temperature.rvs(1)[0])
        # Each time a ball is given a temperature, the maximum bound is subtracted by it
        max_temperature -= ball_temperatures[x]
    ball_temperatures.append(max_temperature)
    # The speed for the ball is given, corresponding to each temperature by: E = 0.5mv**2 = 1.5kT
    ball_speeds = [np.sqrt((3 * (1.38064852 * 10 ** -23) * i) / mass) for i in ball_temperatures]
    ball_velocities = []
    # Each ball has a specific speed, and so its velocity is randomised with a random angle
    for i in range(len(ball_speeds)):
        angle = rn.uniform(-np.pi, np.pi)
        ball_velocities.append([ball_speeds[i] * np.cos(angle), ball_speeds[i] * np.sin(angle)])
    return ball_velocities


# Creates a random list of ball objects, parametrised by:
# number of balls, their mass and radius, the container radius and gas temperature,
# making it flexible
def random_balls(num_balls, radius, mass, temperature, cont_radius):
    balls = []
    pos = []
    vels = velocities_at_given_temp(num_balls, mass, temperature)
    for x in range(num_balls):
        vel = vels[x]
        pos_temp = create_position_in_container(radius, cont_radius)  # Method ensures balls are within container
        balls, new = no_overlap(x, balls, radius, pos_temp, vel, pos, cont_radius, mass)  # Ensures balls don't overlap
        pos.append(new)
    return balls


# Method ensures ball is within the container (constrained by the size of the container and ball radius)
def create_position_in_container(ball_radius, container_radius):
    # Temporary position that is randomly generated
    pos_temp = [rn.uniform(-container_radius + ball_radius, container_radius - ball_radius),
                rn.uniform(-container_radius + ball_radius, container_radius - ball_radius)]
    # Position is tested to be within the constraints
    while np.sqrt(pos_temp[0] ** 2 + pos_temp[1] ** 2) >= (container_radius - ball_radius):
        pos_temp = [rn.uniform(-container_radius + ball_radius, container_radius - ball_radius),
                    rn.uniform(-container_radius + ball_radius, container_radius - ball_radius)]
    return pos_temp


# This method ensures random balls do not overlap
# It uses recursion to decrease the time complexity and hence increase its efficiency
# This method is completely parametrised by the characteristics of the balls and the container
def no_overlap(i, balls, radius, pos_temp, vel, pos, cont_radius, mass):
    if i == 0:  # The first case does not need to compare with others balls (it's still an empty list)
        new_pos = pos_temp
        balls.append(Ball(mass, radius, pos_temp, vel))
        return balls, new_pos
    else:
        distance = []
        for y in range(len(pos)):  # Calculates relative distance with all other existing balls
            distance.append(np.sqrt((pos_temp[0] - pos[y][0]) ** 2 + (pos_temp[1] - pos[y][1]) ** 2))
        if min(distance) > (2 * radius):  # Ensures balls don't overlap (must be at minimum 2 radii away)
            correct = pos_temp
            balls.append(Ball(mass, radius, correct, vel))
            return balls, correct
        else:  # If the balls overlap, the ball is randomly created again, until the condition is met
            new_position = create_position_in_container(radius, cont_radius)
            return no_overlap(i, balls, radius, new_position, vel, pos, cont_radius, mass)  # Recursion  for efficiency
