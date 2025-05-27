import taichi as ti
import numpy as np
import time

# TODO:
# - collision rewrite -> faster
# - bit mask
# - pack the fields together
# - integrate initialization
# - visualization improvements

precision = ti.f32
device = ti.gpu
Length_scale = 2.44e8
Mass_scale = 5e10  # Scale for masses to make them visually significant
Time_scale = 3600 * 1e7  # Scale for time to make the simulation run at a reasonable speed
ti.init(arch=device, default_fp=precision)

# =========================================
# 可视化常数

sun_pos_tuple = (0.5, 0.5, 0.)  # Center of the screen for camera lookat
init_camera_pos = (0.5, 0.5, 2.)  # Camera position for visualization
gui_res = (1600, 1600)
step_per_frame = 6  # Number of steps per frame for smoother and faster animation

# ========================================

# ========================================
# 模拟超参数都在这里

num_particles = 10000  # 星体数量
sun_mass = 1.898 * 1e27  # 中心星体质量
particles_mass_lowerbound = 1e10  # 小行星质量下限
particles_mass_uperbound = 1e12  #小行星质量上线
ring_inner = 1.22e8  #球壳内半径
ring_outer = 1.29e8  #球壳外半径
dt = 60.  # 模拟分辨率，最好别动

# ==================================================

sun_pos = ti.Vector(sun_pos_tuple)  # Center of the screen
sun_mass = sun_mass / Mass_scale  # Scale the sun mass for visualization
particles_mass_lowerbound = particles_mass_lowerbound / Mass_scale
particles_mass_uperbound = particles_mass_uperbound / Mass_scale
particles_mass_width = particles_mass_uperbound - particles_mass_lowerbound
ring_inner = ring_inner / Length_scale  # Scale the ring inner radius for visualization
ring_outer = ring_outer / Length_scale  # Scale the ring outer radius for visualization
ring_center = (ring_inner + ring_outer) / 2
ring_width = ring_outer - ring_inner
dt = dt / Time_scale  # Scale the timestep for visualization
particles_vertices_lowerbound = 15e4
particles_vertices_uperbound = 21e4
particles_vertices_lowerbound = particles_vertices_lowerbound * Time_scale / Length_scale
particles_vertices_uperbound = particles_vertices_uperbound * Time_scale / Length_scale
particles_vertices_width = particles_vertices_uperbound - particles_vertices_lowerbound
G = 6.67430e-11  # Gravitational constant - using a scaled value for visualization
# For visualization purposes, we might need to scale G or masses, or use a different dt
# Let's use a more visual-friendly G
G_viz = G * Time_scale**2 / (Length_scale**3 / Mass_scale)

# radius calculation
radii_coeff = 6e-4  # coefficient for converting mass to radius
sun_radii_coeff = 1e-4  # additional coefficient for sun radius

softening_dis = radii_coeff
scheme = "Verlet"  # "Euler" or "Verlet"
vec = ti.types.vector(3, dtype=float)

# Particle properties
positions = ti.Vector.field(3, dtype=float, shape=num_particles)
positions_stock = ti.Vector.field(3, dtype=float, shape=num_particles)

velocities = ti.Vector.field(3, dtype=float, shape=num_particles)

acceleration = ti.Vector.field(3, dtype=float, shape=num_particles)

masses = ti.field(dtype=float, shape=num_particles)
radii = ti.field(dtype=float, shape=num_particles)


@ti.kernel
def initialize_particles():
	# Initialize Sun
	masses[0] = sun_mass
	positions[0] = sun_pos
	velocities[0] = ti.Vector([0.0, 0.0, 0.0])

	# Initialize other particles
	for i in range(1, num_particles):
		# ===============================================================
		# 初始化函数， 修改这里
		# ToDo:
		# 1. 径向速度
		# 2. 改变分布
		# 3. 环面
		u = ti.random()
		v = ti.random()
		w = ti.random()

		r = ring_width * (u**(1 / 3)) + ring_inner  # 保证球体内均匀分布
		theta = ti.acos(2 * v - 1)  # 极角 [0, π]
		phi = 2 * ti.math.pi * w  # 方位角 [0, 2π]

		pos_x = sun_pos[0] + r * ti.sin(theta) * ti.cos(phi)
		pos_y = sun_pos[1] + r * ti.sin(theta) * ti.sin(phi)
		pos_z = sun_pos[2] + r * ti.cos(theta)

		dx = pos_x - sun_pos[0]
		dy = pos_y - sun_pos[1]
		dz = pos_z - sun_pos[2]

		r_vec = ti.Vector([dx, dy, dz])
		# 任取一个不平行的向量，比如z轴
		ref = ti.Vector([0.0, 0.0, 1.0])  # 先给一个默认值
		if abs(r_vec[0]) < 1e-6 and abs(r_vec[1]) < 1e-6:
			ref = ti.Vector([0.0, 1.0, 0.0])

		tangent = r_vec.cross(ref).normalized()  # cross: 矢量叉乘， normalize：单位向量
		vel_mag = (G_viz * sun_mass / r)**0.5  # 环绕速度大小，不改
		# 切向速度=速度单位矢量*速度大小*随机大小因子
		velocities[i] = tangent * vel_mag * (ti.random() * 0.1 + 0.95)
		#====================================================================
		masses[
		    i] = ti.random() * particles_mass_width + particles_mass_lowerbound
		positions[i] = ti.Vector([pos_x, pos_y, pos_z])
		positions_stock[i] = positions[i]
	update_radii(masses, radii, num_particles)


@ti.func
def calculate_pairwise_acceleration(i, j, positions, masses, G_viz, softening):
	acc_contribution = ti.Vector([0.0, 0.0, 0.0])
	r_vec = positions[j] - positions[i]
	dist = r_vec.norm()  # Squared distance

	if dist > softening:  # only apply force if not too close
		force_dir = r_vec.normalized()
		acc_contribution = G_viz * masses[j] * force_dir / dist**2
	return acc_contribution


@ti.func
def update_radii(all_masses, all_radii, total_num_particles):
	for i in range(total_num_particles):
		all_radii[i] = all_masses[i]**(1 / 3) * radii_coeff

	all_radii[0] = all_radii[0] * sun_radii_coeff


@ti.func
def compute_total_acceleration_on_particle(p_i, all_positions, all_masses,
                                           current_G_viz, total_num_particles,
                                           current_softening):
	acc_sum = ti.Vector([0.0, 0.0, 0.0])
	for p_j in range(num_particles):
		if p_i == p_j:
			continue
		acc_sum += calculate_pairwise_acceleration(p_i, p_j, all_positions,
		                                           all_masses, current_G_viz,
		                                           current_softening)
	return acc_sum


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
	update_radii(masses, radii, num_particles)


# Initialize particles
initialize_particles()

# GUI

window = ti.ui.Window("N-body simulation", gui_res, vsync=True, pos=(50, 50))
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((0., 0., 0.))
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(*init_camera_pos)
camera.up(0, 1, 0)
camera.lookat(*sun_pos_tuple)

# Simulation loop
paused = True
restart_flag = False
# Camera control state variables
orbit_sun_fixed_radius = False  # Renamed from stick_to_large_ball
always_look_at_sun = True  # Default to looking at the sun

while window.running:
	with gui.sub_window("Controls", 0.05, 0.05, 0.2, 0.1):
		button_text = "Resume" if paused else "Pause"
		if gui.button(button_text):
			paused = not paused
		if gui.button("Restart"):
			restart_flag = True

	# Camera Controls GUI
	with gui.sub_window("Camera Controls", 0.05, 0.16, 0.2,
	                    0.12):  # Positioned below "Controls"
		orbit_sun_fixed_radius = gui.checkbox(
		    "Orbit Sun (Radius 2)",
		    orbit_sun_fixed_radius)  # Updated label and variable
		always_look_at_sun = gui.checkbox("Always look at Sun",
		                                  always_look_at_sun)
		if gui.button("Reset Camera"):
			camera.position(*init_camera_pos)
			camera.lookat(*sun_pos_tuple)
			# Reset control states to default
			orbit_sun_fixed_radius = False  # Updated variable
			always_look_at_sun = True

	if restart_flag:
		initialize_particles()
		paused = True  # Pause simulation after restart
		restart_flag = False

	if not paused:
		for i in range(step_per_frame):
			if scheme == "Euler":
				compute_forces()
				update_positions()
				update_collisons()
			elif scheme == "Verlet":
				update_positions_Verlet()
				update_collisons()

	# Draw particles
	# Sun in yellow, others in white

	# Camera logic update
	if orbit_sun_fixed_radius:
		camera.track_user_inputs(
		    window, movement_speed=0.01,
		    hold_key=ti.ui.SHIFT)  # Get user's intended movement/orientation

		current_cam_pos = camera.curr_position  # This is a tuple (x,y,z)
		# sun_pos_tuple is (0.5, 0.5, 0.0)

		# Calculate vector from sun to current camera position
		vec_to_cam_x = current_cam_pos[0] - sun_pos_tuple[0]
		vec_to_cam_y = current_cam_pos[1] - sun_pos_tuple[1]
		vec_to_cam_z = current_cam_pos[2] - sun_pos_tuple[2]

		current_dist_from_sun = (vec_to_cam_x**2 + vec_to_cam_y**2 +
		                         vec_to_cam_z**2)**0.5
		desired_radius = 2.0

		if current_dist_from_sun > 1e-6:  # Avoid division by zero if camera is at sun's center
			# Normalize the vector and scale it to the desired_radius
			norm_factor = desired_radius / current_dist_from_sun
			new_cam_x = sun_pos_tuple[0] + vec_to_cam_x * norm_factor
			new_cam_y = sun_pos_tuple[1] + vec_to_cam_y * norm_factor
			new_cam_z = sun_pos_tuple[2] + vec_to_cam_z * norm_factor
			camera.position(new_cam_x, new_cam_y, new_cam_z)
		else:
			# If camera is at sun's center, place it at a default position on the sphere
			camera.position(sun_pos_tuple[0], sun_pos_tuple[1],
			                sun_pos_tuple[2] + desired_radius)

		if always_look_at_sun:
			camera.lookat(*sun_pos_tuple)
		# If not always_look_at_sun, the look_at direction from track_user_inputs is maintained
		# as camera.position() only changes position, not look_at.
	else:  # orbit_sun_fixed_radius is False
		camera.track_user_inputs(window,
		                         movement_speed=0.01,
		                         hold_key=ti.ui.SHIFT)
		if always_look_at_sun:
			camera.lookat(*sun_pos_tuple)

	# camera.lookat(*sun_pos_tuple) # This line is now handled by the logic above
	scene.ambient_light([.5, .5, .5])
	scene.point_light(pos=(0.5, 0.5, 1.5), color=(1., 1., 1.))
	scene.particles(positions, 0, per_vertex_radius=radii)
	scene.set_camera(camera)
	canvas.scene(scene)
	window.show()
