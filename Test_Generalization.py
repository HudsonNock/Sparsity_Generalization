import torch
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F
from torchrl.envs import GymEnv, ParallelEnv
from torchrl.modules import ProbabilisticActor
from torchrl.data import CompositeSpec
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from tensordict.nn import TensorDictModule
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import InteractionType
from copy import deepcopy # For target networks
import os
import time
import cv2
import numpy as np
import Cave_Flyer as cf
import l0_models

def soft_update(target_net, source_net, tau):
    """Polyak update for target networks."""
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# --- Network Definitions ---

# helper to build a fresh CNN encoder
def make_cnn_encoder():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1), nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2), nn.ReLU(),
        nn.Conv2d(32, 16, 4, stride=2), nn.ReLU(),
        nn.Flatten(),
    )

class CriticNet(nn.Module):
    def __init__(self, encoder: nn.Module, n_flat: int, n_actions: int):
        super().__init__()
        self.encoder = encoder
        self.lin1 = nn.Linear(n_flat, 512)
        self.lin2 = nn.Linear(512, n_actions)

    def forward(self, pixels: torch.Tensor, vel: torch.Tensor, angle: torch.Tensor):
        feats = self.encoder(pixels)
        x = torch.cat([feats, vel, angle], dim=1)
        x = F.relu(self.lin1(x))
        return self.lin2(x)  # [B,1]
    
class PolicyNet(nn.Module):
    def __init__(self, encoder: nn.Module, n_flat: int, n_actions: int):
        super().__init__()
        self.encoder = encoder
        self.lin1 = nn.Linear(n_flat, 512)
        self.lin2 = nn.Linear(512, n_actions)

    def forward(self, pixels: torch.Tensor, vel: torch.Tensor, angle: torch.Tensor):
        feats = self.encoder(pixels)
        x = torch.cat([feats, vel, angle], dim=1)
        x = F.relu(self.lin1(x))
        return self.lin2(x)  # [B,1]


def build_agents_and_buffer(device, load_path, sparse):
    #n_actions = act_spec.shape[-1]
    n_actions = 6

    # Compute flattened feature size automatically
    with torch.no_grad():
        dummy = torch.zeros((1,1,35,35), device=device)
        temp_encoder = make_cnn_encoder().to(device) # Use specified device
        n_flat = temp_encoder(dummy).shape[-1]
        n_flat += 4
        del temp_encoder # clean up

    if sparse:
        policy = l0_models.L0PolicyNet()
    else:
        policy = PolicyNet(make_cnn_encoder(), n_flat, n_actions).to(device) #

    if os.path.exists(load_path):
        try:
            checkpoint = torch.load(load_path, map_location=device) # Load to target device

            def strip_module_prefix(sd):
                return {k.replace("module.", "", 1): v for k, v in sd.items()}

            policy.load_state_dict(strip_module_prefix(checkpoint['actor_state_dict']))

        except Exception as e:
            print(f"Error loading checkpoint from {load_path}: {e}")
            print("Starting training from scratch.")
    else:
        print(f"Checkpoint file not found at {load_path}. Starting training from scratch.")

    policy.training = False

    return policy

def get_prob_from_logits(logits):
    probs = F.softmax(logits, dim=-1)
    return probs

def sample_action_from_probs(probs):
    actions = torch.multinomial(probs, num_samples=1)
    return actions.squeeze()

def main():
    # --- Environment Setup ---
    num_envs = 512
    #env_name = "procgen:procgen-caveflyer"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    #env = ParallelEnv(
    #    num_workers=num_envs,
    #    create_env_fn=lambda: GymEnv(env_name, frame_skip=2, paint_vel_info=True, use_backgrounds=False, num_levels=20, start_level=0),
    #    device='cpu' #device # Can run env stepping on GPU if desired/possible
    #)
    num_seeds = 300 #20
    n_actions = 6
    for j in range(2):
        if j == 0:
            num_seeds = 20
        else:
            num_seeds = 300
        percentage_complete = np.empty((7, num_seeds))
        for i in range(7):
            if j == 0:
                env = cf.VectorizedCaveflyer(num_agents=num_envs, initial_seed=0, num_seeds=num_seeds)
            else:
                env = cf.VectorizedCaveflyer(num_agents=num_envs, initial_seed=2000020, num_seeds=num_seeds)

            # --- Build Agents and Buffer ---
            step_num = int(500*i)
            load_path = f"SACL0_checkpoints2\\checkpoint_010_{step_num}.pth"
            policy = build_agents_and_buffer(device, load_path, True)
            epsilon = 0.05
            max_steps = 300 #500

            # --- Training Loop ---
            pixels, vel, angle = env.reset()
            td = TensorDict({
                "pixels":pixels[:,None,:,:].astype(np.float32) / 250.0,
                "vel":vel.astype(np.float32),
                "angle":angle.astype(np.float32),}, batch_size=[num_envs])
            td = td.to(device)

            print("Starting data collection...")

            cv2.namedWindow("game", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("game", 512, 512)

            steps = np.zeros((num_envs,), dtype=np.int64)
            attempts = np.zeros((num_seeds,), dtype=np.int64)
            successes = np.zeros((num_seeds,), dtype=np.int64)

            num_steps = 1500
            if j == 0:
                num_steps == 500
            for step in range(1, num_steps): # 1000
                player_seed = env.player_seed.copy()
                with torch.no_grad(): # No need for gradients during data collection
                    # Get action from policy
                    logits = policy.forward(td["pixels"].to(device), td["vel"].to(device), td["angle"].to(device)) # Modifies td_current_step in-place adds 'action' and 'logits'
                    probs = get_prob_from_logits(logits)
                    action = sample_action_from_probs(probs)
                    #td["action"] = torch.tensor(np.random.randint(0,8, (32,), dtype=np.int64))
                    random_actions = torch.randint(0, n_actions, (num_envs,), device=td.device, dtype=torch.int64)
                    take_random = torch.rand((num_envs,), device=td.device) < epsilon
                    td["action_idx"] = torch.where(take_random, random_actions, action)
                    # Step the environment
                    pixels, vel, angle, rewards, dones = env.step(np.array(td["action_idx"].to('cpu'))) # Env expects action index
                    # Preprocess next observation
                    steps += 1
                    agents_reset = (steps == max_steps)
                    if np.any(agents_reset):
                        pixels, vel, angle = env.reset(agents_reset)
                        dones[agents_reset] = True
                
                completed_seeds = player_seed[dones]
                successfully_completed_seeds = player_seed[(rewards == 1.0)]
                steps[dones] = 0

                # Increment the attempts for each completed seed
                np.add.at(attempts, completed_seeds, 1)
                np.add.at(successes, successfully_completed_seeds, 1)
                
                # Build the transition TensorDict
                # Keys should match what's expected by the loss calculation later
                transition = TensorDict({
                    "pixels":   td["pixels"].to('cpu'), # Move to CPU for storage
                    "vel": td["vel"].to('cpu'),
                    "angle": td["angle"].to('cpu'),
                    "action": td["action_idx"].to('cpu'),
                    "next": TensorDict({
                        "pixels": torch.tensor(pixels[:,None,:,:].astype(np.float32) / 250.0),
                        "vel": torch.tensor(vel.astype(np.float32)),
                        "angle": torch.tensor(angle.astype(np.float32)),
                        "reward": torch.tensor(rewards.astype(np.float32)),
                        "done":   torch.tensor(dones),
                            # "terminated": td_next_step["next"]["terminated"].cpu() # Include if available/needed
                    }, batch_size=[num_envs], device='cpu'),
                    # Optional: store log_prob if needed, but SAC usually recomputes it
                }, batch_size=[num_envs], device='cpu')

                # Add to replay buffer (handle unbinding for ParallelEnv)

                # Prepare for the next iteration
                td = transition["next"] # Use the 'next' state as the current state

                pixels_np = (td["pixels"][0].detach().to('cpu').numpy() * 255).astype(np.uint8)
                cv2.imshow("game", pixels_np[0,:,:])
                cv2.waitKey(1)
                td = td.to(device)

            not_completed = np.logical_not(dones)

            for step in range(1, max_steps + 10): # Loop based on interactions per env
                if not np.any(not_completed):
                    break
                player_seed = env.player_seed.copy()
                with torch.no_grad(): # No need for gradients during data collection
                    # Get action from policy
                    logits = policy.forward(td["pixels"].to(device), td["vel"].to(device), td["angle"].to(device)) # Modifies td_current_step in-place adds 'action' and 'logits'
                    probs = get_prob_from_logits(logits)
                    action = sample_action_from_probs(probs)
                    #td["action"] = torch.tensor(np.random.randint(0,8, (32,), dtype=np.int64))
                    random_actions = torch.randint(0, n_actions, (num_envs,), device=td.device, dtype=torch.int64)
                    take_random = torch.rand((num_envs,), device=td.device) < epsilon
                    td["action_idx"] = torch.where(take_random, random_actions, action)
                    # Step the environment
                    pixels, vel, angle, rewards, dones = env.step(np.array(td["action_idx"].to('cpu'))) # Env expects action index
                    # Preprocess next observation
                    steps += 1
                    agents_reset = (steps == max_steps)
                    if np.any(agents_reset):
                        pixels, vel, angle = env.reset(agents_reset)
                        dones[agents_reset] = True
                
                completed_seeds = player_seed[np.logical_and(dones, not_completed)]
                successfully_completed_seeds = player_seed[not_completed][(rewards[not_completed] == 1.0)]
                steps[dones] = 0
                if np.any(successfully_completed_seeds == 0):
                    pass

                not_completed = np.logical_and(not_completed, np.logical_not(dones))

                # Increment the attempts for each completed seed
                np.add.at(attempts, completed_seeds, 1)
                np.add.at(successes, successfully_completed_seeds, 1)
                
                # Build the transition TensorDict
                # Keys should match what's expected by the loss calculation later
                transition = TensorDict({
                    "pixels":   td["pixels"].to('cpu'), # Move to CPU for storage
                    "vel": td["vel"].to('cpu'),
                    "angle": td["angle"].to('cpu'),
                    "action": td["action_idx"].to('cpu'),
                    "next": TensorDict({
                        "pixels": torch.tensor(pixels[:,None,:,:].astype(np.float32) / 250.0),
                        "vel": torch.tensor(vel.astype(np.float32)),
                        "angle": torch.tensor(angle.astype(np.float32)),
                        "reward": torch.tensor(rewards.astype(np.float32)),
                        "done":   torch.tensor(dones),
                            # "terminated": td_next_step["next"]["terminated"].cpu() # Include if available/needed
                    }, batch_size=[num_envs], device='cpu'),
                    # Optional: store log_prob if needed, but SAC usually recomputes it
                }, batch_size=[num_envs], device='cpu')

                # Add to replay buffer (handle unbinding for ParallelEnv)

                # Prepare for the next iteration
                td = transition["next"] # Use the 'next' state as the current state

                pixels_np = (td["pixels"][0].detach().to('cpu').numpy() * 255).astype(np.uint8)
                cv2.imshow("game", pixels_np[0,:,:])
                cv2.waitKey(1)
                td = td.to(device)

            for k in range(num_seeds):
                percentage_complete[i,k] = successes[k] / attempts[k]
            print(i)
        if j == 0:
            np.save("results\\SACL0_010_300_Train.npy", percentage_complete)
        else:
            np.save("results\\SACL0_010_300_Test.npy", percentage_complete)


if __name__ == "__main__":
    # this is critical on Windows when using spawn()
    mp.freeze_support()
    # optionally explicitly set start method
    # Using spawn is generally safer for multiprocessing with CUDA
    if torch.cuda.is_available():
        mp.set_start_method("spawn", force=True)
    main()