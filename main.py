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
init_step_per_frame = 6  # Number of steps per frame for smoother and faster animation
simu_fps = 30  # frames per second for simulation mode

# ========================================

# ========================================
# 模拟超参数都在这里

init_num_particles = 10000  # 星体数量
sun_mass = 1.898 * 1e27  # 中心星体质量
particles_mass_lowerbound = 1e10  # 小行星质量下限
particles_mass_uperbound = 1e12  #小行星质量上线
ring_inner = 1.22e8  #球壳内半径
ring_outer = 1.29e8  #球壳外半径
dt = 60.  # 模拟分辨率，最好别动
compact_rate = 0.5  # 压缩率，压缩率越大，压缩越频繁

# ==================================================

step_per_frame = init_step_per_frame  # Number of steps per frame for smoother animation
num_particles = init_num_particles  # Number of particles in the simulation
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
def initialize_particles(num_particles: int):
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
	for p_j in range(total_num_particles):
		if p_i == p_j:
			continue
		acc_sum += calculate_pairwise_acceleration(p_i, p_j, all_positions,
		                                           all_masses, current_G_viz,
		                                           current_softening)
	return acc_sum


@ti.kernel
def compute_forces(num_particles: int):
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
def update_positions(num_particles: int):
	for i in range(num_particles):
		if i == 0:  # Sun is fixed
			continue
		positions[i] += velocities[i] * dt


@ti.kernel
def update_positions_Verlet(num_particles: int):

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
def update_collisons(num_particles: int):
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


@ti.kernel
def compact_particles_kernel(current_num_particles: int) -> int:
	# current_num_particles is the current total number of particles, including the sun.
	# The sun at index 0 is always active and stays.
	write_idx = 1  # Start writing active non-sun particles from index 1

	# Iterate over non-sun particles
	ti.loop_config(serialize=True)  # Ensure the loop is serialized for safety
	for read_idx in range(1, current_num_particles):
		if masses[read_idx] > 1e-9:  # Active particle threshold
			if read_idx != write_idx:
				# Copy data for active particle to the new 'write_idx'
				positions[write_idx] = positions[read_idx]
				velocities[write_idx] = velocities[read_idx]
				masses[write_idx] = masses[read_idx]
				radii[write_idx] = radii[read_idx]
				positions_stock[write_idx] = positions_stock[read_idx]
				acceleration[write_idx] = acceleration[read_idx]
			write_idx += 1  # Increment for the next active particle

	# Zero out the masses and other properties of particles that are no longer in the active list
	# This ranges from the new count of active particles (write_idx) up to the old count (current_num_particles)
	for i in range(write_idx, current_num_particles):
		masses[i] = 0.0
		velocities[i] = ti.Vector([0.0, 0.0, 0.0])
		radii[i] = 0.0
		positions_stock[i] = ti.Vector([0.0, 0.0, 0.0])
		acceleration[i] = ti.Vector([0.0, 0.0, 0.0])

	return write_idx  # This is the new count of total particles (sun is at 0, active particles from 1 to write_idx-1)


# Initialize particles
initialize_particles(num_particles)

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
simulation = False  # Flag for accelerating simulation
# Camera control state variables
orbit_sun_fixed_radius = False  # Default to not orbiting the sun with a fixed radius
always_look_at_sun = True  # Default to looking at the sun

# num_particles = 5000

last_time = time.time()  # For FPS calculation
while window.running:

	with gui.sub_window("Controls", 0.05, 0.05, 0.2,
	                    0.12):  # Adjusted height from 0.1 to 0.12
		button_text = "Resume" if paused else "Pause"
		if gui.button(button_text):
			paused = not paused
		if gui.button("Restart"):
			restart_flag = True
		simulation = gui.checkbox("Simulation Mode", simulation)

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

	# Calculate particle statistics for display
	masses_np = masses.to_numpy()
	# Exclude the sun (particle 0) from statistics
	non_sun_masses_np = masses_np[1:num_particles]
	# Consider particles with mass > 1e-9 (a small threshold) as active
	active_particles_mask = non_sun_masses_np > 1e-9
	active_masses_values = non_sun_masses_np[active_particles_mask]
	num_active_particles = len(active_masses_values)

	top_particles_info = []
	if num_active_particles > 0:
		total_mass_val = np.sum(active_masses_values)
		# Sort active masses in descending order and get their values
		sorted_active_masses = np.sort(active_masses_values)[::-1]

		for i in range(min(10, num_active_particles)):
			mass_val = sorted_active_masses[i]
			percentage = (mass_val /
			              total_mass_val) * 100 if total_mass_val > 0 else 0
			top_particles_info.append((mass_val, percentage))
	else:
		total_mass_val = 0.0

	# If active particles are less than half of the total particles,
	# we copy the non-zero mass particles to the front of fields
	# then we decrease num_particles
	if num_particles > 1 and num_active_particles < (num_particles -
	                                                 1) * compact_rate:
		new_total_particles = compact_particles_kernel(num_particles)
		if new_total_particles < num_particles and new_total_particles > 0:  # Ensure it decreased and is valid
			num_particles = new_total_particles
			# print(f"Compaction occurred. New num_particles: {num_particles}") # Optional debug

	# Particle Statistics GUI
	with gui.sub_window("Particle Statistics", 0.05, 0.29, 0.2,
	                    0.3):  # x, y, width, height
		gui.text(f"Active Particles: {num_active_particles}")
		gui.text(f"Total Mass: {total_mass_val:.2e}")
		gui.text("Top 10 Particles by Mass:")
		if num_active_particles > 0:
			for i, (mass_val, percentage) in enumerate(top_particles_info):
				gui.text(f"  {i+1}. Mass: {mass_val:.2e} ({percentage:.2f}%)")
		else:
			gui.text("  No active particles.")

	if restart_flag:
		num_particles = init_num_particles
		initialize_particles(num_particles)
		paused = True  # Pause simulation after restart
		restart_flag = False
		simulation = False  # Reset simulation mode
		step_per_frame = init_step_per_frame  # Reset steps per frame

	if not paused:
		for i in range(step_per_frame):
			if scheme == "Euler":
				compute_forces(num_particles)
				update_positions(num_particles)
				update_collisons(num_particles)
			elif scheme == "Verlet":
				update_positions_Verlet(num_particles)
				update_collisons(num_particles)

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

	current_time = time.time()
	delta_time = current_time - last_time
	if delta_time > 1e-9:
		fps = 1.0 / delta_time
	else:
		fps = simu_fps  # Default to max_fps if delta_time is zero or too small
	last_time = current_time

	# FPS and step_per_frame adjustment logic
	if simulation:
		# Try to increase steps if FPS is comfortably above simu_fps
		if fps > (simu_fps * 1.1 + 1):
			step_per_frame *= (fps / simu_fps) / 1.1
			step_per_frame = int(step_per_frame)
		elif fps > simu_fps + 1:
			# Increase steps if FPS is above simu_fps but not too high
			step_per_frame += 1
		# Decrease steps if FPS is lower than simu_fps, but not below init_step_per_frame
		elif fps < simu_fps and step_per_frame > init_step_per_frame:
			step_per_frame -= 1

		# Ensure step_per_frame is at least init_step_per_frame when accelerating
		step_per_frame = max(init_step_per_frame, step_per_frame)
	else:
		# Reset to initial steps per frame if acceleration is off
		step_per_frame = init_step_per_frame
