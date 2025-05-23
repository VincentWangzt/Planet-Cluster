import taichi as ti

precision = ti.f32
device = ti.cpu

ti.init(arch=device, default_fp=precision)

num_particles = 10000
dt = 1e-6  # Timestep
G = 6.67430e-11  # Gravitational constant - using a scaled value for visualization
# For visualization purposes, we might need to scale G or masses, or use a different dt
# Let's use a more visual-friendly G
G_viz = 1e-2

scheme = "Verlet"  # "Euler" or "Verlet"

# Particle properties
positions = ti.Vector.field(2, dtype=float, shape=num_particles)
positions_stock = ti.Vector.field(2, dtype=float, shape=num_particles)
visible_positions = ti.Vector.field(2, dtype=float, shape=num_particles)

velocities = ti.Vector.field(2, dtype=float, shape=num_particles)
# velocities_stock = ti.Vector.field(2, dtype=float, shape=num_particles)

# acceleration_stock = ti.Vector.field(2, dtype=float, shape=num_particles)
acceleration = ti.Vector.field(2, dtype=float, shape=num_particles)

masses = ti.field(dtype=float, shape=num_particles)
radii = ti.field(dtype=float, shape=num_particles)
visible_radii = ti.field(dtype=float, shape=num_particles)

# Sun properties
sun_mass = 1e9
sun_pos = ti.Vector([0.5, 0.5])  # Center of the screen

vec = ti.types.vector(2, dtype=float)

radii_coeff = 3e-4  # coefficient for converting mass to radius
sun_radii_coeff = 1e-1  # additional coefficient for sun radius

# plotted_radius = (visible_radii_coeff * radius + 1) ** visible_radii_power
visible_radii_coeff = 3e2  # coefficient for plotting the radius
visible_radii_power = 1  # power for plotting the radius

softening_dis = radii_coeff


@ti.func
def calculate_pairwise_acceleration(i, j, positions, masses, G_viz,
                                    softening) -> vec:
	acc_contribution = ti.Vector([0.0, 0.0])
	r_vec = positions[j] - positions[i]
	dist = r_vec.norm()  # Squared distance

	if dist > softening:  # only apply force if not too close
		force_dir = r_vec.normalized()
		acc_contribution = G_viz * masses[j] * force_dir / dist**2
	return acc_contribution


@ti.func
def update_radii(all_masses, all_radii, visible_radii, total_num_particles):
	for i in range(total_num_particles):
		all_radii[i] = all_masses[i]**(1 / 3) * radii_coeff
		visible_radii[i] = (visible_radii_coeff * all_radii[i] +
		                    1)**visible_radii_power * ti.math.sign(
		                        all_masses[i])

	all_radii[0] = all_radii[0] * sun_radii_coeff
	visible_radii[0] = (visible_radii_coeff * all_radii[0] +
	                    1)**visible_radii_power


@ti.func
def compute_total_acceleration_on_particle(p_i, all_positions, all_masses,
                                           current_G_viz, total_num_particles,
                                           current_softening) -> vec:
	acc_sum = ti.Vector([0.0, 0.0])
	for p_j in range(total_num_particles):
		if p_i == p_j:
			continue
		acc_sum += calculate_pairwise_acceleration(p_i, p_j, all_positions,
		                                           all_masses, current_G_viz,
		                                           current_softening)
	return acc_sum


@ti.kernel
def initialize_particles():
	# Initialize Sun
	masses[0] = sun_mass
	positions[0] = sun_pos
	velocities[0] = ti.Vector([0.0, 0.0])

	# Initialize other particles
	for i in range(1, num_particles):
		r = ti.random() * 0.4 + 0.1  # Distance from sun (0.1 to 0.5)
		angle = ti.random() * 2 * ti.math.pi
		pos_x = sun_pos[0] + r * ti.cos(angle)
		pos_y = sun_pos[1] + r * ti.sin(angle)
		positions[i] = ti.Vector([pos_x, pos_y])
		positions_stock[i] = positions[i]

		# Give some initial orbital velocity
		# Velocity perpendicular to the vector from sun to particle
		# Magnitude chosen to be somewhat stable, can be tuned
		dist_sq = r**2
		if dist_sq > 1e-6:  # avoid division by zero if particle is at sun's position
			# Circular orbit velocity: v = sqrt(G * M_sun / r)
			# We use G_viz here
			vel_mag = (G_viz * sun_mass / r)**0.5
			# Direction perpendicular to (pos - sun_pos)
			# If (dx, dy) is vector from sun, perpendicular is (-dy, dx) or (dy, -dx)
			dx = pos_x - sun_pos[0]
			dy = pos_y - sun_pos[1]
			velocities[i] = ti.Vector([-dy, dx]).normalized() * vel_mag * (
			    ti.random() * 0.1 + 0.95
			)  # factor to make orbits more dynamic/less stable initially
		else:
			velocities[i] = ti.Vector([0.0, 0.0])
		# velocities_stock[i] = velocities[i]
		# velocities[i] = ti.Vector([0.0, 0.0])
		masses[i] = ti.randn() * 1. + 3.
		update_radii(masses, radii, visible_radii, num_particles)


@ti.kernel
def compute_forces():
	# The sun is fixed, so we don't update its velocity based on other particles
	# But other particles affect each other and are affected by the sun
	for i in range(num_particles):
		if i == 0:  # Skip the sun for force calculation *on* it, as it's fixed
			continue

		# Softening factor to prevent extreme forces at close distances
		softening = 1e-3
		total_acc_on_i = compute_total_acceleration_on_particle(
		    i, positions, masses, G_viz, num_particles, softening)
		velocities[i] += total_acc_on_i * dt


@ti.kernel
def update_positions():
	for i in range(num_particles):
		if i == 0:  # Sun is fixed
			continue
		positions[i] += velocities[i] * dt

		# # Boundary conditions (optional: particles can escape or wrap around)
		# # Simple reflection for now
		# if positions[i][0] < 0 or positions[i][0] > 1:
		# 	velocities[i][0] *= -0.8  # Lose some energy on bounce
		# 	positions[i][0] = ti.max(0, ti.min(1, positions[i][0]))
		# if positions[i][1] < 0 or positions[i][1] > 1:
		# 	velocities[i][1] *= -0.8  # Lose some energy on bounce
		# 	positions[i][1] = ti.max(0, ti.min(1, positions[i][1]))


@ti.kernel
def update_positions_Verlet():

	for i in range(num_particles):
		if i == 0:
			continue

		# Update positions using the Verlet method
		tmp = positions[i]
		positions[i] = positions[
		    i] + velocities[i] * dt + 0.5 * acceleration[i] * dt**2
		positions_stock[i] = tmp

	for i in range(num_particles):
		if i == 0:
			continue
		# Update velocities using the average of the current and previous accelerations
		tmp = acceleration[i]
		acceleration[i] = compute_total_acceleration_on_particle(
		    i, positions, masses, G_viz, num_particles, softening_dis)
		velocities[i] = velocities[i] + 0.5 * (acceleration[i] + tmp) * dt


@ti.kernel
def update_collisons():
	for i in range(num_particles):
		for j in range(i + 1, num_particles):
			if (positions[i] - positions[j]).norm() < (radii[i] + radii[j]):
				# Handle collision
				# For simplicity, we can just merge the particles
				masses[i] += masses[j]
				positions[i] = (positions[i] * masses[i] + positions[j] *
				                masses[j]) / (masses[i] + masses[j])
				velocities[i] = (velocities[i] * masses[i] + velocities[j] *
				                 masses[j]) / (masses[i] + masses[j])
				# radii[i] = radii[i] + radii[j]
				# Remove j particle
				masses[j] = 0
	update_radii(masses, radii, visible_radii, num_particles)


# Initialize particles
initialize_particles()

# GUI
gui_res = (800, 800)
gui = ti.GUI("N-body Simulation", res=gui_res, background_color=0x000000)

# Simulation loop
while gui.running:
	# for _ in range(2):  # Substeps for stability
	if scheme == "Euler":
		compute_forces()
		update_positions()
		update_collisons()
	elif scheme == "Verlet":
		update_positions_Verlet()
		update_collisons()

	# Draw particles
	# Sun in yellow, others in white
	gui.circles(positions.to_numpy()[1:],
	            radius=visible_radii.to_numpy()[1:],
	            color=0xFFFFFF)
	gui.circles(positions.to_numpy()[0:1],
	            radius=visible_radii.to_numpy()[0:1],
	            color=0xFFD700)  # Sun

	gui.show()
