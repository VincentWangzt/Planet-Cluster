import taichi as ti
import numpy as np
import time
import argparse  # Added import

# TODO:
# - collision debug
# + distance visualization
# - change radii coeff -> density
# + add number of particles within a certain distance
# - argument parser
# + total time

precision = ti.f32
device = ti.gpu
# Length_scale = 2.44e8 # Will be replaced by args
# Mass_scale = 5e10  # Scale for masses to make them visually significant # Will be replaced by args
# Time_scale = 3600 * 1e7  # Scale for time to make the simulation run at a reasonable speed # Will be replaced by args
ti.init(arch=device, default_fp=precision)

# =========================================
# Command line arguments setup
parser = argparse.ArgumentParser(description="N-body simulation with Taichi")
parser.add_argument('--gui_res_width',
                    type=int,
                    default=1600,
                    help='GUI window width')
parser.add_argument('--gui_res_height',
                    type=int,
                    default=1600,
                    help='GUI window height')
parser.add_argument('--init_step_per_frame',
                    type=int,
                    default=6,
                    help='Initial number of simulation steps per frame')
parser.add_argument('--init_num_particles',
                    type=int,
                    default=10000,
                    help='Initial number of particles')
parser.add_argument(
    '--sun_mass_val',
    type=float,
    default=1.898e27,
    help='Mass of the central star (e.g., Sun)')  # Renamed to avoid conflict
parser.add_argument('--original_dt_value',
                    type=float,
                    default=60.0,
                    help='Simulation time step in seconds (model time)')
parser.add_argument('--compact_rate',
                    type=float,
                    default=0.5,
                    help='Compaction rate for particle removal')
parser.add_argument(
    '--dist_active_threshold_sim_units',
    type=float,
    default=5.0,
    help='Distance threshold for dist-active particles in simulation units')
parser.add_argument('--length_scale',
                    type=float,
                    default=2.44e8,
                    help='Length scale for simulation units')
parser.add_argument('--mass_scale',
                    type=float,
                    default=5e10,
                    help='Mass scale for simulation units')
parser.add_argument('--time_scale',
                    type=float,
                    default=3600 * 1e7,
                    help='Time scale for simulation units')
parser.add_argument(
    '--radii_coeff',
    type=float,
    default=6e-4,
    help='Coefficient for converting mass to radius for particles')
parser.add_argument('--sun_radii_coeff',
                    type=float,
                    default=1e-4,
                    help='Additional coefficient for sun radius')
parser.add_argument('--scheme',
                    type=str,
                    default="Verlet",
                    choices=['Euler', 'Verlet'],
                    help='Integration scheme (Euler or Verlet)')

args = parser.parse_args()

# Use parsed arguments
Length_scale = args.length_scale
Mass_scale = args.mass_scale
Time_scale = args.time_scale
# =========================================
# 可视化常数

sun_pos_tuple = (0.5, 0.5, 0.)  # Center of the screen for camera lookat
init_camera_pos = (0.5, 0.5, 2.)  # Camera position for visualization
gui_res = (args.gui_res_width, args.gui_res_height)  # Use parsed args
init_step_per_frame = args.init_step_per_frame  # Use parsed args
init_target_fps = 30  # Target FPS for simulation and display modes

# ========================================

# ========================================
# 模拟超参数都在这里

init_num_particles = args.init_num_particles  # 星体数量 # Use parsed args
sun_mass_val = args.sun_mass_val  # 中心星体质量 # Use parsed args, renamed from sun_mass to avoid conflict later
particles_mass_lowerbound = 1e10  # 小行星质量下限
particles_mass_uperbound = 1e12  #小行星质量上线
ring_inner = 1.22e8  #球壳内半径
ring_outer = 1.29e8  #球壳外半径
original_dt_value = args.original_dt_value  # 模拟分辨率 (seconds per simulation step in model time) # Use parsed args
compact_rate = args.compact_rate  # 压缩率，压缩率越大，压缩越频繁 # Use parsed args
DIST_ACTIVE_THRESHOLD_SIM_UNITS = args.dist_active_threshold_sim_units  # Distance threshold for dist-active particles in simulation units # Use parsed args

# ==================================================

unscaled_dt_value = original_dt_value
step_per_frame = init_step_per_frame  # Number of steps per frame for smoother animation
target_fps = init_target_fps  # Initialize target_fps
num_particles = init_num_particles  # Number of particles in the simulation
sun_pos = ti.Vector(sun_pos_tuple)  # Center of the screen
sun_mass = args.sun_mass_val / Mass_scale  # Scale the sun mass for visualization # Use sun_mass_val
particles_mass_lowerbound = particles_mass_lowerbound / Mass_scale
particles_mass_uperbound = particles_mass_uperbound / Mass_scale
particles_mass_width = particles_mass_uperbound - particles_mass_lowerbound
ring_inner = ring_inner / Length_scale  # Scale the ring inner radius for visualization
ring_outer = ring_outer / Length_scale  # Scale the ring outer radius for visualization
ring_center = (ring_inner + ring_outer) / 2
ring_width = ring_outer - ring_inner
dt = unscaled_dt_value / Time_scale  # Scale the timestep for simulation
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
radii_coeff = args.radii_coeff  # coefficient for converting mass to radius # Use parsed args
sun_radii_coeff = args.sun_radii_coeff  # additional coefficient for sun radius # Use parsed args

softening_dis = radii_coeff
scheme = args.scheme  # "Euler" or "Verlet" # Use parsed args
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
def update_collisons(num_particles: int) -> bool:
	flag = False
	# ti.loop_config(serialize=True)  # Ensure the loop is serialized for safety
	for i in range(num_particles):
		for j in range(i + 1, num_particles):
			if (positions[i] - positions[j]).norm() < (radii[i] + radii[j]):
				# Handle collision
				if masses[i] != 0 and masses[j] != 0:
					flag = True
				# For simplicity, we can just merge the particles
				positions[i] = (positions[i] *
				                (masses[i] + 1e-6) + positions[j] *
				                (masses[j] + 1e-6)) / (masses[i] + masses[j] +
				                                       2e-6)
				velocities[i] = (velocities[i] *
				                 (masses[i] + 1e-6) + velocities[j] *
				                 (masses[j] + 1e-6)) / (masses[i] + masses[j] +
				                                        2e-6)
				masses[i] += masses[j]
				masses[j] = 0
	update_radii(masses, radii, num_particles)
	return flag


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
display_mode = False  # Flag for new display mode
# Camera control state variables
orbit_sun_fixed_radius = False  # Default to not orbiting the sun with a fixed radius
always_look_at_sun = True  # Default to looking at the sun

# num_particles = 5000
fps = 1
total_simulated_time = 0.0  # Added: Initialize total simulated time

last_time = time.time()  # For FPS calculation
while window.running:

	with gui.sub_window(
	    "Controls", 0.02, 0.05, 0.23,
	    0.25):  # Adjusted x from 0.05 to 0.02, w from 0.2 to 0.23
		button_text = "Resume" if paused else "Pause"
		if gui.button(button_text):
			paused = not paused
		if gui.button("Restart"):
			restart_flag = True

		# Store previous checkbox states to detect user clicks
		prev_simulation_checkbox_state = simulation
		new_simulation_val = gui.checkbox("Simulation Mode", simulation)

		prev_display_mode_checkbox_state = display_mode
		new_display_mode_val = gui.checkbox("Display Mode", display_mode)

		# Handle changes initiated by user clicking checkboxes for mutual exclusivity
		if new_simulation_val != prev_simulation_checkbox_state:  # User clicked simulation checkbox
			simulation = new_simulation_val
			if simulation:  # If simulation was turned ON
				display_mode = False  # Turn off display mode if simulation is on
				target_fps = init_target_fps  # Reset target_fps for simulation mode
		elif new_display_mode_val != prev_display_mode_checkbox_state:  # User clicked display_mode checkbox
			display_mode = new_display_mode_val
			if display_mode:  # If display_mode was turned ON
				simulation = False  # Turn off simulation mode if display mode is on
				step_per_frame = init_step_per_frame  # Reset steps per frame for display mode
				target_fps = init_target_fps  # Reset target_fps for display mode

		if simulation or display_mode:
			target_fps = gui.slider_int("Target FPS", target_fps, 10, 30)

		if display_mode:
			gui.text("Display Mode Controls:")
			step_per_frame = gui.slider_int(
			    "Steps Per Frame", step_per_frame, 1,
			    20)  # Max value 20 as per user's file
			# unscaled_dt_value slider removed for display mode

	# Camera Controls GUI
	with gui.sub_window(
	    "Camera Controls",
	    0.02,  # Adjusted x from 0.05 to 0.02
	    0.31,
	    0.23,  # Adjusted w from 0.2 to 0.23
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
	positions_np = positions.to_numpy()
	sun_actual_pos_np = positions_np[0]  # Sun's current position

	# Consider particles with mass > 1e-9 (a small threshold) as active
	# Exclude the sun (particle 0) from statistics initially
	non_sun_masses_np = masses_np[1:num_particles]
	non_sun_positions_np = positions_np[1:num_particles]

	active_particles_mask = non_sun_masses_np > 1e-9
	active_masses_values = non_sun_masses_np[active_particles_mask]
	active_positions_values = non_sun_positions_np[active_particles_mask]

	num_active_particles = len(active_masses_values)

	# Calculate dist-active particles (non-sun particles within a distance threshold from the sun)
	num_dist_active_particles = 0
	if num_active_particles > 1:  # If there are any non-sun particles
		# These are the positions of all non-sun particles currently in the simulation
		distances_from_sun = np.linalg.norm(active_positions_values -
		                                    sun_actual_pos_np,
		                                    axis=1)
		num_dist_active_particles = np.sum(
		    distances_from_sun < DIST_ACTIVE_THRESHOLD_SIM_UNITS)

	top_particles_info = []

	if num_active_particles > 0:
		# Sort active masses in descending order and get their original indices within the active_masses_values array
		sorted_indices_of_active = np.argsort(active_masses_values)[::-1]

		total_mass_val = np.sum(active_masses_values)

		for i in range(min(10, num_active_particles)):
			# Get the index in the sorted list
			sorted_idx = sorted_indices_of_active[i]

			# Get mass and position of this particle
			mass_val = active_masses_values[sorted_idx]
			pos_val = active_positions_values[sorted_idx]

			distance_to_sun = np.linalg.norm(pos_val - sun_actual_pos_np)
			percentage = (mass_val /
			              total_mass_val) * 100 if total_mass_val > 0 else 0
			top_particles_info.append((mass_val, percentage, distance_to_sun))
	else:
		total_mass_val = 0.0

	# If active particles are less than half of the total particles,
	# we copy the non-zero mass particles to the front of fields
	# then we decrease num_particles
	if num_particles > 1 and num_active_particles < (num_particles -
	                                                 1) * compact_rate:
		new_total_particles_auto = compact_particles_kernel(num_particles)
		if new_total_particles_auto < num_particles and new_total_particles_auto > 0:  # Ensure it decreased and is valid
			num_particles = new_total_particles_auto
			# print(f"Auto compaction. New num_particles: {num_particles}") # Optional debug

	# Particle Statistics GUI
	with gui.sub_window(
	    "Particle Statistics", 0.02, 0.44, 0.23,
	    0.38):  # Adjusted x from 0.05 to 0.02, w from 0.2 to 0.23
		gui.text(f"Active Particles: {num_particles}")
		gui.text(f"Effective Particles: {num_active_particles}")
		gui.text(
		    f"Dist-Active (<{DIST_ACTIVE_THRESHOLD_SIM_UNITS:.1f} units): {num_dist_active_particles}"
		)  # Added
		if gui.button("Compact Particles Manually"):
			if num_particles > 1:  # Ensure there's more than just the sun
				new_compacted_num = compact_particles_kernel(num_particles)
				if new_compacted_num < num_particles and new_compacted_num > 0:
					num_particles = new_compacted_num
					# print(f"Manual compaction. New num_particles: {num_particles}")
		gui.text(f"Sun Mass: {masses_np[0]:.2e}")
		gui.text(f"Total Mass: {total_mass_val:.2e}")
		gui.text("Top 10 Particles by Mass:")
		if num_active_particles > 0:
			for i, (mass_val, percentage,
			        distance_val) in enumerate(top_particles_info):
				gui.text(
				    f"  {i+1}. Mass: {mass_val:.2e} ({percentage:.2f}%) Dist: {distance_val*Length_scale:.2e} (m)"
				)
		else:
			gui.text("  No active particles.")

	# Simulation Info GUI
	with gui.sub_window(
	    "Simulation Info",
	    0.02,  # Adjusted x from 0.05 to 0.02
	    0.80,
	    0.23,  # Adjusted w from 0.2 to 0.23
	    0.16
	):  # Height remains 0.14 -> Adjusted to 0.16 to make space for new text
		gui.text(f"Target FPS: {target_fps}")
		gui.text(f"Actual FPS: {fps:.2f}")  # Added Actual FPS display
		gui.text(f"Step per frame: {step_per_frame}")
		time_speed_up_factor = unscaled_dt_value * step_per_frame * fps * int(
		    not paused)  # Calculate based on original dt
		gui.text(f"Time Speed Up: {time_speed_up_factor:.2f}x")
		gui.text(f"Total Time Simulated: {total_simulated_time:.2e}s")  # Added

	if restart_flag:
		num_particles = init_num_particles
		initialize_particles(num_particles)
		paused = True  # Pause simulation after restart
		restart_flag = False
		simulation = False  # Reset simulation mode
		display_mode = False  # Reset display mode
		step_per_frame = init_step_per_frame  # Reset steps per frame
		target_fps = init_target_fps  # Reset target_fps
		total_simulated_time = 0.0  # Added: Reset total_simulated_time

	if not paused:
		for i in range(step_per_frame):
			if scheme == "Euler":
				compute_forces(num_particles)
				update_positions(num_particles)
				while update_collisons(num_particles):
					pass
			elif scheme == "Verlet":
				update_positions_Verlet(num_particles)
				while update_collisons(num_particles):
					pass
		total_simulated_time += unscaled_dt_value * step_per_frame  # Added: Increment total_simulated_time

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

	# Sleep logic for Display Mode to try and match target_fps
	if display_mode and not paused and target_fps > 0:
		# last_time is the timestamp from the beginning of the current frame's processing
		# time.time() is the current time, after window.show()
		desired_frame_duration = 1.0 / target_fps
		actual_elapsed_time_for_frame = time.time(
		) - last_time  # Time spent so far in this frame

		sleep_duration = desired_frame_duration - actual_elapsed_time_for_frame * 2
		if sleep_duration > 0:
			time.sleep(sleep_duration)

	current_time = time.time()
	delta_time = current_time - last_time
	if delta_time > 1e-9:
		fps = 1.0 / delta_time
	else:
		fps = target_fps  # Default to target_fps if delta_time is zero or too small
	last_time = current_time

	# FPS and step_per_frame adjustment logic
	if simulation and not paused:  # display_mode is implicitly False here due to mutual exclusivity
		# Try to increase steps if FPS is comfortably above target_fps
		if fps > (target_fps * 1.1 + 1):  # Use target_fps
			step_per_frame *= (fps / target_fps) / 1.1  # Use target_fps
			step_per_frame = int(step_per_frame)
		elif fps > target_fps + 1:  # Use target_fps
			# Increase steps if FPS is above target_fps but not too high
			step_per_frame += 1
		# Decrease steps if FPS is lower than target_fps, but not below init_step_per_frame
		elif fps < target_fps and step_per_frame > init_step_per_frame:  # Use target_fps
			step_per_frame -= 1

		# Ensure step_per_frame is at least init_step_per_frame when accelerating
		step_per_frame = max(init_step_per_frame, step_per_frame)
	else:
		# Reset to initial steps per frame if acceleration is off or paused,
		# but only if not in display_mode (which has its own step_per_frame control)
		if not display_mode:
			step_per_frame = init_step_per_frame
