import numpy as np
import random
import cv2
import time
import pygame
from scipy.ndimage import label

class VectorizedCaveflyer:
    def __init__(self, num_agents=1, map_size=(20, 20), render_size=35,
                 initial_seed=0, num_seeds=20):
        self.num_agents = num_agents
        self.map_size = np.array(map_size)
        self.render_size = render_size

        self.initial_global_seed = initial_seed
        self.num_seeds = num_seeds

        # Initialize agent-specific seeds. These will be updated.
        self.agent_current_seeds = np.zeros(num_agents, dtype=int)
        for i in range(num_agents):
            # Derive initial seed for each agent predictably
            self.agent_current_seeds[i] = self.initial_global_seed + random.randint(0,num_seeds-1)


        self.dt = 1
        self.rotation_omega = np.pi/7.0
        self.forward_acceleration = 0.1
        self.num_actions = 6 # 0:no-op, 1:left, 2:right, 3:thrust, 4:thrust_left, 5:thrust_right

        self.shades = {
            "background": 0, "wall": 150, "player": 200,
            "asteroid": 75, "goal": 250
        }

        # Data structures for multiple independent environments

        self.maps = np.ones((num_seeds, self.map_size[0], self.map_size[1]), dtype=np.uint8)
        self.initial_player_pos = np.empty((num_seeds, 2)) # Store initial pos for resets
        self.goal_pos = np.empty((num_seeds,2))
        # List of arrays, as asteroid counts can vary per agent/map

        self.positions = np.zeros((num_agents, 2))
        self.velocities = np.zeros((num_agents, 2))
        self.angles = np.zeros(num_agents)
        self.closest_distances_to_goal = np.zeros(num_agents)
        self.player_seed = np.zeros((num_agents), dtype=int)

        self.initalize_maps()

    def _get_new_seed_idx_for_done_agent(self):
        return random.randint(0, self.num_seeds-1)
        
    def initalize_maps(self):
        for i in range(self.num_seeds):
            self._generate_cave(i, self.initial_global_seed + i)

        for i in range(self.num_agents):
            self.player_seed[i] = random.randint(0,self.num_seeds-1)

    def _generate_cave(self, seed_idx, seed_to_use):
        # Temporarily seed numpy and random for this agent's map generation
        temp_np_state = np.random.get_state()
        temp_random_state = random.getstate()

        np.random.seed(seed_to_use)
        random.seed(seed_to_use)

        num_nodes = random.randint(2,4)
        cave_map = np.full(self.map_size, self.shades["wall"], dtype=np.uint8)

        rng = np.random.default_rng(seed=seed_to_use)

        # Step 2: Generate num_nodes distinct points and store them in a list
        points_list = []
        # Use a set for efficient checking of duplicate coordinates
        occupied_coords = set() 
        pos_goal = (0,0)
        init_p_pos = (0,0)

        if num_nodes > 0:
            while len(points_list) < num_nodes:
                row, col = rng.integers(1, self.map_size[0]-1, size=2)
                if len(points_list) == num_nodes-1:
                    init_p_pos = points_list[0]
                    self.initial_player_pos[seed_idx] = [points_list[0][0] + 0.5, points_list[0][1] + 0.5]
                    pos_goal = (row, col)
                    if np.linalg.norm(np.array(init_p_pos) - np.array(pos_goal)) > self.map_size.mean() * 0.2:
                        occupied_coords.add((row, col))
                        points_list.append((row, col))
                elif (row, col) not in occupied_coords:
                    occupied_coords.add((row, col))
                    points_list.append((row, col))
        
        # Step 3: Draw lines between adjacent points in the list
        # This requires at least 2 nodes to form a path.
        for i in range(num_nodes - 1):
            p1_row, p1_col = points_list[i]
            p2_row, p2_col = points_list[i+1] # The next point in the sequence

            # Draw Manhattan line: Horizontal from p1, then Vertical to p2's column ending at p2's row
            
            # Horizontal segment: from (p1_row, p1_col) to (p1_row, p2_col)
            min_c, max_c = min(p1_col, p2_col), max(p1_col, p2_col)
            cave_map[p1_row, min_c : max_c + 1] = self.shades["background"]
            
            # Vertical segment: from (p1_row, p2_col) to (p2_row, p2_col)
            min_r, max_r = min(p1_row, p2_row), max(p1_row, p2_row)
            cave_map[min_r : max_r + 1, p2_col] = self.shades["background"]
    
        # Step 4: Iterative Expansion
        # Max iterations safeguard: n*n is the theoretical max cells to flip one by one.
        # Add a small buffer, e.g., +10 for the loop counter itself.
        perimiter = np.full(cave_map.shape, False, dtype=np.bool_)
        perimiter[1:len(perimiter)-1, 1:len(perimiter[0])-1] = True

        num_iterations = random.randint(80,160)
        for _ in range(num_iterations): 
            # Identify cells with value 1 that are orthogonally adjacent to any cell with value 0
            is_zero_mask = (cave_map == self.shades["background"])
            
            padded_is_zero_mask = np.pad(is_zero_mask, pad_width=1, mode='constant', constant_values=False)

            neighbor_up_is_zero    = padded_is_zero_mask[ :-2, 1:-1]
            neighbor_down_is_zero  = padded_is_zero_mask[2:  , 1:-1]
            neighbor_left_is_zero  = padded_is_zero_mask[1:-1,  :-2]
            neighbor_right_is_zero = padded_is_zero_mask[1:-1, 2:  ]
            
            has_zero_neighbor_mask = (neighbor_up_is_zero | neighbor_down_is_zero |
                                    neighbor_left_is_zero | neighbor_right_is_zero)
            
            candidate_mask = (cave_map == self.shades["wall"]) & has_zero_neighbor_mask & perimiter
            candidate_coords = np.argwhere(candidate_mask)

            if candidate_coords.shape[0] == self.shades["background"]:
                # No more 1s are adjacent to 0s, so expansion naturally stops
                break
            
            chosen_candidate_idx_in_list = rng.integers(0, candidate_coords.shape[0])
            r_flip, c_flip = candidate_coords[chosen_candidate_idx_in_list]
            
            cave_map[r_flip, c_flip] = self.shades["background"]

        wall_mask = (cave_map == self.shades["wall"])

        # label connected components (4-connectivity)
        structure = np.array([[0,1,0],
                            [1,1,1],
                            [0,1,0]], dtype=int)
        labeled_array, num_features = label(wall_mask, structure=structure)

        # mask of border pixels
        border_mask = np.zeros_like(cave_map, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True

        # find which labels touch the border
        border_labels = np.unique(labeled_array[border_mask])

        # all labels minus those touching the border
        all_labels = np.arange(1, num_features + 1)
        internal_labels = np.setdiff1d(all_labels, border_labels)

        # return a list of masks, one per internal island
        islands = [(labeled_array == lbl) for lbl in internal_labels]

        for island_mask in islands:
            cave_map[island_mask] = self.shades["background"]

        cave_map[pos_goal] = self.shades["goal"]

        self.goal_pos[seed_idx] = [pos_goal[0] + 0.5, pos_goal[1] + 0.5]

        def find_empty_spot(current_map_for_agent):
            while True:
                pos = np.random.rand(2) * (self.map_size - 2) + 1
                if current_map_for_agent[int(pos[0]), int(pos[1])] == self.shades["background"]:
                    return pos.astype(int)

        cave_map[init_p_pos] = self.shades["player"]
        num_asteroids = np.random.randint(3, 8)
        for _ in range(num_asteroids):
            ast_pos = find_empty_spot(cave_map)
            cave_map[ast_pos[0], ast_pos[1]] = self.shades["asteroid"]
        cave_map[init_p_pos] = self.shades["background"]
        self.maps[seed_idx] = cave_map

        # Restore global random state
        np.random.set_state(temp_np_state)
        random.setstate(temp_random_state)


    def _reset_agent_sub_environment(self, agent_idx, seed_idx):
        self.player_seed[agent_idx] = seed_idx

        self.positions[agent_idx] = self.initial_player_pos[seed_idx]
        self.velocities[agent_idx] = np.zeros(2)
        self.angles[agent_idx] = random.uniform(0, 2 * np.pi) # Use global random for initial angle
        self.closest_distances_to_goal[agent_idx] = np.linalg.norm(self.positions[agent_idx] - self.goal_pos[self.player_seed[agent_idx]])

    def reset(self, agent_ids=None):
        if agent_ids is None:
            agent_ids_to_reset = np.arange(self.num_agents)
        elif agent_ids.dtype == np.bool_:
            agent_ids_to_reset = np.where(agent_ids)[0]
        else:
            agent_ids_to_reset = np.array(agent_ids, dtype=int)

        for i in agent_ids_to_reset:
            self._reset_agent_sub_environment(i, random.randint(0, self.num_seeds-1))
        return self._get_obs()

    def _get_obs(self):
        # This function should render the current state for each agent,
        # even if their episode just ended and they were auto-reset.
        # The self.dones flag inside _get_obs is not used to blank image if auto-resetting.
        pixels = np.empty((self.num_agents, self.render_size, self.render_size), dtype=np.uint8)
        vel = np.empty((self.num_agents, 2))
        angle = np.empty((self.num_agents, 2))

        for i in range(self.num_agents):
            # Even if self.dones[i] is True from a *previous* step and this is an external reset,
            # we generate a fresh observation. If it was auto-reset, self.dones[i] is already False.
            map_cpy = self.maps[self.player_seed[i]].copy()
            map_cpy = map_cpy.repeat(2, axis=0).repeat(2, axis=1)

            map_center_row, map_center_col = np.clip(int(self.positions[i][0]*2), 0, len(map_cpy)-1), np.clip(int(self.positions[i][1]*2), 0, len(map_cpy[0])-1)

            map_cpy[map_center_row][map_center_col] = self.shades["player"]

            half = self.render_size // 2

            # Define the ranges in original array
            i_min = map_center_row - half + 1
            i_max = map_center_row + half
            j_min = map_center_col - half + 1
            j_max = map_center_col + half

            # Create padded array
            padded = np.full((self.render_size, self.render_size), self.shades["wall"], dtype=map_cpy.dtype)

            # Compute valid bounds
            i_start = max(i_min, 0)
            i_end = min(i_max, len(map_cpy)-1)
            j_start = max(j_min, 0)
            j_end = min(j_max, len(map_cpy[0])-1)

            # Compute where to place this in the padded array
            pi_start = i_start - i_min
            pi_end = pi_start + (i_end - i_start)
            pj_start = j_start - j_min
            pj_end = pj_start + (j_end - j_start)

            # Copy the data
            padded[pi_start:pi_end, pj_start:pj_end] = map_cpy[i_start:i_end, j_start:j_end]

            pixels[i] = padded
            vel[i] = self.velocities[i]
            angle[i] = np.array([np.cos(self.angles[i]), np.sin(self.angles[i])])
        return pixels, vel, angle

    def get_obs_vectorized(self):
        n = self.positions.shape[0]
        # stack & upsample maps
        flat_maps = np.stack(self.maps, axis=0)           # (M,H,W)
        agent_maps = flat_maps[self.player_seed]                # (n,H,W)
        maps2 = np.repeat(np.repeat(agent_maps, 2, axis=1), 2, axis=2)  # (n,2H,2W)

        # compute center indices
        H2, W2 = maps2.shape[1:]
        center_r = np.clip((self.positions[:,0]*2).astype(int), 0, H2-1)
        center_c = np.clip((self.positions[:,1]*2).astype(int), 0, W2-1)

        # mark player
        idx = np.arange(n)
        maps2[idx, center_r, center_c] = self.shades['player']

        # pad by half window
        half = self.render_size // 2
        pad_maps = np.pad(
            maps2,
            ((0,0),(half,half),(half,half)),
            mode='constant', constant_values=self.shades['wall']
        )  # (n,2H+2half,2W+2half)

        # relative offsets
        offs = np.arange(-half+1, half+1)
        # build per-agent window indices
        pr = center_r + half  # shifted into pad
        pc = center_c + half
        row_idx = pr[:,None,None] + offs[None,None,:]  # (n,1,render)
        col_idx = pc[:,None,None] + offs[None,:,None]  # (n,render,1)

        # extract with advanced indexing
        pixels = pad_maps[
            idx[:,None,None],   # (n,1,1)
            row_idx,            # (n,1,render)
            col_idx             # (n,render,1)
        ]  # broadcasts to (n,render,render)

        # velocities & angles
        vel = self.velocities.copy()
        angle = np.stack([np.cos(self.angles), np.sin(self.angles)], axis=1)

        return pixels, vel, angle

    def step(self, actions):
        # convert to numpy array
        actions = np.array(actions, dtype=int)
        n = self.num_agents

        # map actions to thrusts and rotations via boolean masks
        thrust = np.zeros(n)
        rotation = np.zeros(n)
        mask = (actions == 1)
        rotation[mask] = -self.rotation_omega
        mask = (actions == 2)
        rotation[mask] =  self.rotation_omega
        mask = (actions == 3)
        thrust[mask]  =  self.forward_acceleration
        mask = (actions == 4)
        thrust[mask]  =  self.forward_acceleration
        rotation[mask] = -self.rotation_omega
        mask = (actions == 5)
        thrust[mask]  =  self.forward_acceleration
        rotation[mask] =  self.rotation_omega

        # update angles and velocities
        self.angles = (self.angles + rotation * self.dt) % (2 * np.pi)
        self.velocities *= 0.9
        # apply thrust where needed
        self.velocities[:,0] += thrust * np.sin(self.angles) * self.dt
        self.velocities[:,1] += thrust * np.cos(self.angles) * self.dt

        # clamp speeds
        #speeds = np.linalg.norm(self.velocities, axis=1)
        #too_fast = speeds > 0.3
        #if np.any(too_fast):
        #    self.velocities[too_fast,:] = (self.velocities[too_fast,:] * np.tile((0.3 / speeds[too_fast])[:,None], (1,2)))

        # predict new positions
        new_pos = self.positions + self.velocities * self.dt

        # handle collisions per axis via vectorized helper
        """
        pos_x, vel_x = zip(*[self._collide_axis(0, self.positions[i], new_pos[i], self.velocities[i],
                                                 self.maps[self.player_seed[i]])
                              for i in range(n)])
        pos_x = np.vstack(pos_x)
        vel_x = np.vstack(vel_x)

        pos_xy, vel_xy = zip(*[self._collide_axis(1, pos_x[i], new_pos[i], vel_x[i],
                                                   self.maps[self.player_seed[i]])
                                 for i in range(n)])
        self.positions = np.vstack(pos_xy)
        self.velocities = np.vstack(vel_xy)
        """

        # vectorized wall collisions: first X axis, then Y axis
        pos_x, vel_x = self.collide_axis_vectorized(
            axis=0,
            old_pos=self.positions,
            new_pos=new_pos,
            vel=self.velocities,
        )
        pos_xy, vel_xy = self.collide_axis_vectorized(
            axis=1,
            old_pos=pos_x,
            new_pos=new_pos,
            vel=vel_x,
        )
        old_pos = self.positions.copy()
        self.positions = pos_xy
        self.velocities = vel_xy

        # rewards and done flags
        done = np.zeros(n, dtype=bool)
        rewards = np.zeros(n)

        # 1) integer grid locations of agents
        pos_int = self.positions.astype(int)         # (n,2)
        
        # 2) 3×3 offsets around each integer position
        offs = np.array([[-1,-1],[-1,0],[-1,1],
                        [ 0,-1],[ 0,0],[ 0,1],
                        [ 1,-1],[ 1,0],[ 1,1]], dtype=int)  # (9,2)
        
        # 3) for each agent, for each offset, compute tile coords
        #    shape (n,9,2)
        tile_coords = pos_int[:,None,:] + offs[None,:,:]
        
        # 4) clamp to valid map bounds if you need to treat out‐of‐bounds as non‐asteroid
        H, W = self.maps[0].shape
        tile_coords_clipped = np.empty_like(tile_coords)
        tile_coords_clipped[...,0] = np.clip(tile_coords[...,0], 0, H-1)
        tile_coords_clipped[...,1] = np.clip(tile_coords[...,1], 0, W-1)
        
        # 5) gather the map values for each agent
        #    we first duplicate the agent‐specific map index to shape (n,9)
        seeds = self.player_seed[:,None]                         # (n,1)
        seeds_rep = np.repeat(seeds, offs.shape[0], axis=1)  # (n,9)
        
        # 6) get the map‐values: 
        #    we flatten everything so we can fancy‐index in one go
        flat_maps = np.stack(self.maps, axis=0)  # (M, H, W)
        tx = tile_coords_clipped[...,0].ravel()  # (n*9,)
        ty = tile_coords_clipped[...,1].ravel()  # (n*9,)
        ts = seeds_rep.ravel()                   # (n*9,)
        tiles_flat = flat_maps[ts, tx, ty]       # (n*9,)
        tiles = tiles_flat.reshape(n, 9)         # (n,9)
        
        # 7) find which of those 9 are asteroids
        is_asteroid = (tiles == self.shades["asteroid"])  # (n,9)
        
        # 8) compute the (n,9,2) coords of the asteroid centers
        #     integer tile + offset + 0.5
        asteroid_centers = tile_coords.astype(float) + 0.5  # (n,9,2)
        
        # 9) compute distances to agent centers
        #    subtract positions[:,None,:], square, sum, sqrt
        deltas = self.positions[:,None,:] - asteroid_centers  # (n,9,2)
        d2 = np.sum(deltas**2, axis=2)                   # (n,9)
        d = np.sqrt(d2)                                  # (n,9)
        
        # 10) test within collision radius (0.25 + 0.5)
        collision = is_asteroid & (d < (0.25 + 0.5))  # (n,9)
        
        # 11) an agent dies if any of its 9 offsets collide and it wasn’t already done
        done = np.any(collision, axis=1)  # (n,)

        # vectorized goal distances
        goal_coords = self.goal_pos[self.player_seed]
        dists_to_goal = np.linalg.norm(self.positions - goal_coords, axis=1)
        hit_goal = dists_to_goal < (0.25 + 0.5)
        rewards[hit_goal & ~done] = 1.0
        done[hit_goal] = True

        # distance-based incremental rewards
        dist_improved = (dists_to_goal < self.closest_distances_to_goal) & ~done
        reward_scale = 4.0 / self.map_size.mean()
        rewards[dist_improved] += (self.closest_distances_to_goal[dist_improved] - dists_to_goal[dist_improved]) * reward_scale
        self.closest_distances_to_goal[dist_improved] = dists_to_goal[dist_improved]

        #reset envs
        num_resets = done.sum()
        self.player_seed[done] = np.random.randint(0, self.num_seeds, size=(num_resets,))

        self.positions[done] = self.initial_player_pos[self.player_seed[done]]
        self.velocities[done] = np.zeros(2)
        self.angles[done] = np.random.uniform(0, 2 * np.pi, size=(num_resets,)) # Use global random for initial angle
        self.closest_distances_to_goal[done] = np.linalg.norm(self.positions[done] - self.goal_pos[self.player_seed[done]], axis=1)

        pixels, vel, angle = self._get_obs()
        return pixels, vel, angle, rewards, done

    """
    def step(self, actions):
        actions = np.array(actions, dtype=int)
        current_rewards = np.zeros(self.num_agents)
        # This will be the 'done' flags returned, indicating episode termination THIS step
        done = np.full((self.num_agents), False, dtype=np.bool_)

        for i in range(self.num_agents):
            action = actions[i]
            thrust = 0; rotation = 0
            if action == 1: rotation = -self.rotation_omega
            elif action == 2: rotation = self.rotation_omega
            elif action == 3: thrust = self.forward_acceleration
            elif action == 4: thrust = self.forward_acceleration; rotation = -self.rotation_omega
            elif action == 5: thrust = self.forward_acceleration; rotation = self.rotation_omega

            self.angles[i] = (self.angles[i] + rotation * self.dt) % (2 * np.pi)
            self.velocities[i] *= 0.9
            if thrust > 0:
                self.velocities[i, 0] += thrust * np.sin(self.angles[i]) * self.dt
                self.velocities[i, 1] += thrust * np.cos(self.angles[i]) * self.dt

            if np.linalg.norm(self.velocities[i]) > 1.2:
                self.velocities[i] = self.velocities[i] / np.linalg.norm(self.velocities[i]) * 1.2
            new_pos = self.positions[i] + self.velocities[i] * self.dt

            current_agent_map = self.maps[self.player_seed[i]]

            pos_x, vel_x = self._collide_axis(
                axis=0,
                old_pos=self.positions[i],
                new_pos=new_pos,
                vel=self.velocities[i],
                game_map=current_agent_map
            )
            # then Y
            pos_xy, vel_xy = self._collide_axis(
                axis=1,
                old_pos=pos_x,
                new_pos=new_pos,
                vel=vel_x,
                game_map=current_agent_map
            )

            self.positions[i] = pos_xy
            self.velocities[i] = vel_xy

            pos_int = self.positions[i].astype(int)

            # Check for termination conditions
            for er in range(-1,2):
                for ec in range(-1,2):
                    if 0 <= pos_int[0] + er < len(current_agent_map) and 0 < pos_int[1] + ec < len(current_agent_map):
                        if current_agent_map[pos_int[0] + er, pos_int[1] + ec] == self.shades["asteroid"] and not done[i]:
                            pos_asteroid = np.array([pos_int[0] + er + 0.5, pos_int[1] + ec + 0.5])
                            if np.linalg.norm(self.positions[i] - pos_asteroid) < 0.25 + 0.5:
                                current_rewards[i] = 0
                                done[i] = True
            
            goal_pos_f = self.goal_pos[self.player_seed[i]]
            
            if not done[i]:
                if np.linalg.norm(self.positions[i] - goal_pos_f) < 0.25 + 0.5:
                    current_rewards[i] = 1.0
                    done[i] = True

            if not done[i]: # Distance-based reward if not terminated
                current_dist_to_goal = np.linalg.norm(self.positions[i] - goal_pos_f)
                if current_dist_to_goal < self.closest_distances_to_goal[i]:
                    reward_scale = 4.0 / (self.map_size.mean())
                    current_rewards[i] += (self.closest_distances_to_goal[i] - current_dist_to_goal) * reward_scale
                    self.closest_distances_to_goal[i] = current_dist_to_goal

            if done[i]:
                new_seed = self._get_new_seed_idx_for_done_agent()
                self._reset_agent_sub_environment(i, new_seed)

        pixels, vel, angle = self._get_obs() # Get observations AFTER any auto-resets

        # `self.dones` is now False for any agent that was auto-reset, ready for the next step.
        return pixels, vel, angle, current_rewards, done
    """

    import numpy as np

    def collide_axis_vectorized(self, axis, old_pos, new_pos, vel):
        n = old_pos.shape[0]
        H, W = self.map_size

        # tentative positions
        pos_tent = old_pos.copy()
        pos_tent[:, axis] = new_pos[:, axis]

        # integer base
        pos_int = pos_tent.astype(int)  # (n,2)

        # 3x3 offsets
        offs = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])

        # tile indices
        tiles = pos_int[:, None, :] + offs[None, :, :]
        tiles[...,0] = np.clip(tiles[...,0], 0, H-1)
        tiles[...,1] = np.clip(tiles[...,1], 0, W-1)

        # gather map values
        flat_maps = np.stack(self.maps, axis=0)
        seeds_r = np.repeat(self.player_seed[:, None], 9, axis=1).ravel()
        tx = tiles[...,0].ravel()
        ty = tiles[...,1].ravel()
        vals = flat_maps[seeds_r, tx, ty].reshape(n, 9)

        # wall mask
        wall_mask = (vals == self.shades['wall'])

        # closest points
        # for each tile, clip pos_tent[...] between tile and tile+1
        base = pos_tent[:, None, :]
        txf = tiles.astype(float)
        closest = np.clip(base, txf, txf+1)  # (n,9,2)

        # distances squared
        d2 = np.sum((base - closest)**2, axis=2)

        # collision if wall & within radius
        coll = wall_mask & (d2 < 0.25**2)
        any_coll = np.any(coll, axis=1)

        # apply reversion
        out_pos = pos_tent.copy()
        out_vel = vel.copy()
        idx = np.where(any_coll)[0]
        out_pos[idx, axis] = old_pos[idx, axis]
        out_vel[idx, axis] = 0.0

        return out_pos, out_vel


    def _collide_axis(self, axis, old_pos, new_pos, vel, game_map):
        # axis: 0 for x, 1 for y
        # create a tentative position copy
        pos = old_pos.copy()
        pos[axis] = new_pos[axis]

        # sample points around circular hitbox in this axis direction
        # we'll step along movement direction
        # bounding box of hit circle around new pos
        min_x = pos[0] - 0.25
        max_x = pos[0] + 0.25
        min_y = pos[1] - 0.25
        max_y = pos[1] + 0.25

        # determine tile indices to check
        x0 = max(int(min_x), 0)
        x1 = min(int(max_x), self.map_size[0] - 1)
        y0 = max(int(min_y), 0)
        y1 = min(int(max_y), self.map_size[1] - 1)

        # check each tile for collision
        for tx in range(x0, x1 + 1):
            for ty in range(y0, y1 + 1):
                if game_map[tx, ty] == self.shades['wall']:
                    # closest point in tile to circle center
                    closest_x = np.clip(pos[0], tx, tx + 1)
                    closest_y = np.clip(pos[1], ty, ty + 1)
                    dist_sq = (pos[0] - closest_x)**2 + (pos[1] - closest_y)**2
                    if dist_sq < 0.25**2:
                        # collision: revert this axis and zero velocity
                        pos[axis] = old_pos[axis]
                        vel[axis] = 0
                        return pos, vel
        return pos, vel

if __name__ == '__main__':
    # Use a fixed initial seed for the set of environments,
    # but let done agents pick new random seeds.
    # Or provide a list: done_seeds_list=[1000, 2000, 3000]
    num_test_agents = 512
    env = VectorizedCaveflyer(num_agents=num_test_agents, initial_seed=20, num_seeds=1)

    cv2.namedWindow("game", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("game", 512, 512)

    obs = env.reset()


    pygame.init()
    # small dummy window (required to get events)
    screen = pygame.display.set_mode((200, 200))
    pygame.display.set_caption("Control the first ship with arrows")
    clock = pygame.time.Clock()

    KEY2ACTION = {
        pygame.K_LEFT:  1,  # rotate left
        pygame.K_RIGHT: 2,  # rotate right
        pygame.K_UP:    3,  # thrust forward
        pygame.K_DOWN:  0,  # you could map down to no-op or reverse-thrust
    }

    for step_count in range(1000): # Simulate a few "logical" episodes or sequences of steps
        # read current key state
        keys = pygame.key.get_pressed()

        # decide first-agent action
        action0 = 0  # default = do nothing
        for key, act in KEY2ACTION.items():
            if keys[key]:
                action0 = act
                break

        # build full action list
        # first agent controlled by arrow keys, others random
        num_test_agents = env.num_agents
        actions = [action0] + [
            random.randint(0, env.num_actions - 1)
            for _ in range(num_test_agents - 1)
        ]

        # step your env
        pixels, vel, angle, rewards, dones = env.step(actions)
        
        #actions = [random.randint(0, env.num_actions - 1) for _ in range(num_test_agents)]
        #pixels, vel, angle, rewards, dones_returned = env.step(actions)

        #print(rewards)
        cv2.imshow("game", pixels[0])
        cv2.waitKey(1)
        #time.sleep(0.1)
        