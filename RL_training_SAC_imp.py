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


def build_agents_and_buffer(device, sparse, load_path, lmbda):
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
        policy = l0_models.L0PolicyNet(lambas=(lmbda, lmbda, lmbda, lmbda))
    else:
        policy = PolicyNet(make_cnn_encoder(), n_flat, n_actions).to(device)

    # 2. Define Critic Networks (Two Q-networks for SAC)
    q_net1 = CriticNet(make_cnn_encoder(), n_flat, n_actions).to(device)
    q_net2 = CriticNet(make_cnn_encoder(), n_flat, n_actions).to(device)

    qvalue1_td = TensorDictModule(
        module=q_net1,
        in_keys=["pixels", "vel", "angle"], # Expect one-hot action
        out_keys=["state_action_value1"],
    ).to(device)

    qvalue2_td = TensorDictModule(
        module=q_net2,
        in_keys=["pixels", "vel", "angle"], # Expect one-hot action
        out_keys=["state_action_value2"],
    ).to(device)

    # 3. Define Target Q-Networks
    target_qvalue1_td = deepcopy(qvalue1_td).to(device)
    target_qvalue2_td = deepcopy(qvalue2_td).to(device)
    # Freeze target networks
    for p in target_qvalue1_td.parameters():
        p.requires_grad = False
    for p in target_qvalue2_td.parameters():
        p.requires_grad = False
    target_qvalue1_td.eval()
    target_qvalue2_td.eval()

    # 4. Define Temperature Alpha (learnable)
    # Initialize log_alpha (more stable optimization)
    log_alpha = torch.tensor([-4.1], requires_grad=True, device=device) # Initial alpha = exp(0) = 1.0

    # 5. Create a replay buffer
    buffer_size = 100_000
    batch_size  = 128
    storage = LazyTensorStorage(
        max_size=buffer_size,
        device=torch.device("cpu") # Store buffer data on CPU to save GPU memory
    )
    replay_buffer = TensorDictReplayBuffer(
        storage=storage,
        sampler=RandomSampler(),
        batch_size=batch_size,
    )

    # 6. Set up optimizers (separate for actor, critics, alpha)
    actor_optim = torch.optim.Adam(policy.parameters(), lr=1e-4)
    critic1_optim = torch.optim.Adam(qvalue1_td.parameters(), lr=8e-4)
    critic2_optim = torch.optim.Adam(qvalue2_td.parameters(), lr=8e-4)
    alpha_optim = torch.optim.Adam([log_alpha], lr=3e-4) # Optimizer for log_alpha

    if load_path is not None and os.path.exists(load_path):
        try:
            checkpoint = torch.load(load_path, map_location=device) # Load to target device

            def strip_module_prefix(sd):
                return {k.replace("module.", "", 1): v for k, v in sd.items()}

            policy.load_state_dict(strip_module_prefix(checkpoint['actor_state_dict']))
            q_net1.load_state_dict(strip_module_prefix(checkpoint['critic1_state_dict']))
            q_net2.load_state_dict(strip_module_prefix(checkpoint['critic2_state_dict']))
            target_qvalue1_td.load_state_dict(checkpoint['target1_state_dict'])
            target_qvalue2_td.load_state_dict(checkpoint['target2_state_dict'])

            actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
            critic1_optim.load_state_dict(checkpoint['critic1_optim_state_dict'])
            critic2_optim.load_state_dict(checkpoint['critic2_optim_state_dict'])
            alpha_optim.load_state_dict(checkpoint['alpha_optim_state_dict'])

            # Load log_alpha value from checkpoint (overwrites initial value)
            # Make sure the loaded tensor has requires_grad=True and is on the correct device
            # log_alpha is managed by its optimizer, but we can load the value explicitly if saved
            if 'log_alpha' in checkpoint:
                 # Re-assigning might break the link with the optimizer if not careful.
                 # It's often better to rely on the optimizer loading the state.
                 # However, if you saved it explicitly and want to load:
                 with torch.no_grad():
                      loaded_log_alpha = checkpoint['log_alpha'].to(device).requires_grad_(True)
                      # If log_alpha is a parameter managed directly (not just via optim):
                      log_alpha.copy_(loaded_log_alpha)
                      # Since log_alpha IS the parameter in the optimizer list, loading the
                      # optimizer state should correctly restore it. Let's trust the optimizer load.
                      pass # Trusting optimizer load for log_alpha

            # **Crucially, update target networks to match loaded Q-networks**
            #target_qvalue1_td.load_state_dict(qvalue1_td.state_dict())
            #target_qvalue2_td.load_state_dict(qvalue2_td.state_dict())
            # Ensure target nets are in eval mode after loading state
            target_qvalue1_td.eval()
            target_qvalue2_td.eval()

            # Optional: Load total_frames to resume counting if needed in main()
            # total_frames_loaded = checkpoint.get('total_frames', 0)
            # This would require returning total_frames_loaded from this function

            print(f"Successfully loaded checkpoint from {load_path}")

        except Exception as e:
            print(f"Error loading checkpoint from {load_path}: {e}")
            print("Starting training from scratch.")
    else:
        print(f"Checkpoint file not found at {load_path}. Starting training from scratch.")

    return (policy, qvalue1_td, qvalue2_td, target_qvalue1_td, target_qvalue2_td,
            log_alpha, replay_buffer, actor_optim, critic1_optim, critic2_optim, alpha_optim)

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
    lambdas = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

    #env = ParallelEnv(
    #    num_workers=num_envs,
    #    create_env_fn=lambda: GymEnv(env_name, frame_skip=2, paint_vel_info=True, use_backgrounds=False, num_levels=20, start_level=0),
    #    device='cpu' #device # Can run env stepping on GPU if desired/possible
    #)
    for lmbda in lambdas:
        env = cf.VectorizedCaveflyer(num_agents=num_envs, initial_seed=0, num_seeds=20)

        n_actions = 6

        # --- Build Agents and Buffer ---
        (policy, q_net1, q_net2, target_q_net1, target_q_net2, log_alpha,
        replay_buffer, actor_optim, critic1_optim, critic2_optim, alpha_optim
        ) = build_agents_and_buffer(device, sparse= (lmbda != 0), load_path=None, lmbda=lmbda)

        # --- Hyperparameters ---
        gamma = 0.99           # Discount factor
        tau = 0.005            # Target network polyak averaging factor
        target_entropy = 0.8 #-torch.log(torch.tensor(1.0 / n_actions)).item() # Target entropy heuristic
        collect_steps  = 6       # Collect transitions for X steps before potentially updating
        batch_size     = 128
        initial_collect_frames = 2000 # Collect some random frames before starting training
        epsilon = 0.05

        # --- Training Loop ---
        #td = env.reset()
        # Preprocess initial observation
        #td["pixels"] = (torch.relu(td["pixels"][:,:,:,1]*1.2/255 - td["pixels"][:,:,:,2]*0.6/255))[:,None,:,:]#torch.mean(td["pixels"], dim=-1, dtype=torch.float32)[:, None, :, :]
        pixels, vel, angle = env.reset()
        td = TensorDict({
            "pixels":pixels[:,None,:,:].astype(np.float32) / 250.0,
            "vel":vel.astype(np.float32),
            "angle":angle.astype(np.float32),}, batch_size=[num_envs])
        td = td.to(device)
        total_frames = 0

        print(f"Starting data collection... lambda = {lmbda}")

        train = True

        steps = np.zeros((num_envs,), dtype=np.int64)
        max_steps = 500

        if not train:
            cv2.namedWindow("game", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("game", 512, 512)

        for step in range(0, 1201): # Loop based on interactions per env
            # --- Collect Transitions ---
            for _ in range(collect_steps):
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

                    steps += 1
                    steps[dones] = 0

                    agents_reset = (steps == max_steps)
                    if np.any(agents_reset):
                        pixels, vel, angle = env.reset(agents_reset)
                        dones[agents_reset] = True
                        steps[agents_reset] = 0
                    # Preprocess next observation
                    #td["next"]["pixels"] = (torch.relu(td["next"]["pixels"][:,:,:,1]*1.2/255 - td["next"]["pixels"][:,:,:,2]*0.6/255))[:,None,:,:] #torch.mean(td["next"]["pixels"], dim=-1, dtype=torch.float32)[:, None, :, :]
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
                for i, single_td in enumerate(transition.unbind(0)):   # this gives you 8 TensorDicts of batch_size [1]
                    if not agents_reset[i]:
                        replay_buffer.add(single_td)
                # Update total frames collected
                total_frames += num_envs

                # Prepare for the next iteration
                td = transition["next"] # Use the 'next' state as the current state

                if not train:
                    print(td["reward"].detach().to('cpu').numpy()[0])
                    time.sleep(0.03)
                    pixels_np = (td["pixels"][0].detach().to('cpu').numpy() * 255).astype(np.uint8)
                    cv2.imshow("game", pixels_np[0,:,:])
                    cv2.waitKey(1)
                    pass
                td = td.to(device)


            # --- Update Networks ---
            if total_frames >= initial_collect_frames and train:
                for j in range(int(1.5*collect_steps*num_envs/batch_size)):
                    # Sample a batch
                    if len(replay_buffer) < batch_size:
                        continue # Don't train if buffer isn't full enough

                    batch = replay_buffer.sample(batch_size=batch_size).to(device) # Move batch to training device

                    # Prepare data from batch
                    pixels = batch["pixels"]
                    vel = batch["vel"]
                    angle = batch["angle"]
                    action = batch["action"] # Use one-hot action
                    next_pixels = batch["next"]["pixels"]
                    next_vel = batch["next"]["vel"]
                    next_angle = batch["next"]["angle"]
                    reward = batch["next"]["reward"]
                    done = batch["next"]["done"]
                    # Ensure reward and done have correct shape [batch_size, 1]
                    reward = reward.view(-1, 1)
                    done = done.view(-1, 1).float() # Ensure float for calculations

                    alpha = log_alpha.exp() # Current temperature value

                    # --- Critic Loss (Q-Loss) ---
                    with torch.no_grad(): # Target calculations don't need gradients
                        # Get next action distribution from policy
                        next_logits = policy.forward(next_pixels, next_vel, next_angle) # Get next logits
                        next_probs = F.softmax(next_logits, dim=-1)
                        next_log_probs = F.log_softmax(next_logits, dim=-1)

                        next_td = TensorDict({"pixels": next_pixels, "vel": next_vel, "angle": next_angle}, batch_size=batch.batch_size)
                        target_q1_all_actions = target_q_net1(next_td)["state_action_value1"]
                        target_q2_all_actions = target_q_net2(next_td)["state_action_value2"]

                        # Take the minimum of the two target Q networks
                        min_target_q_all_actions = torch.min(target_q1_all_actions, target_q2_all_actions)

                        # Calculate expected target value V(s') = sum_a'[pi(a'|s') * (Q_target_min(s', a') - alpha * log pi(a'|s'))]
                        next_state_value = torch.sum(next_probs * (min_target_q_all_actions - alpha.detach() * next_log_probs), dim=-1, keepdim=True)

                        # Calculate Bellman target y = r + gamma * (1 - d) * V(s')
                        target_q_value = reward + gamma * (1.0 - done) * next_state_value

                    # Calculate current Q values from main networks for the actions taken
                    current_td = TensorDict({"pixels": pixels, "vel": vel, "angle": angle}, batch_size=batch.batch_size)
                    current_q1 = q_net1(current_td)["state_action_value1"].gather(1, action.unsqueeze(1)).squeeze(1)
                    current_q2 = q_net2(current_td)["state_action_value2"].gather(1, action.unsqueeze(1)).squeeze(1)
                    
                    # Calculate Critic loss (MSE)
                    loss_q1 = F.mse_loss(current_q1, target_q_value.detach().squeeze())
                    loss_q2 = F.mse_loss(current_q2, target_q_value.detach().squeeze())
                    loss_q = loss_q1 + loss_q2

                    # Optimize Critics
                    critic1_optim.zero_grad()
                    loss_q1.backward()
                    critic1_optim.step()

                    critic2_optim.zero_grad()
                    loss_q2.backward()
                    critic2_optim.step()

                    # Get current action distribution from policy for the batch states
                    current_logits = policy.forward(pixels, vel, angle) # Get current logits
                    current_probs = F.softmax(current_logits, dim=-1)
                    current_log_probs = F.log_softmax(current_logits, dim=-1)

                    # Calculate Q values for all actions using the *current* policy and *main* Q networks
                    # Reuse expanded/flattened tensors from critic loss if possible, but recompute Q-values

                    q1_all_actions = target_q_net1(current_td)["state_action_value1"]
                    q2_all_actions = target_q_net2(current_td)["state_action_value2"]
                    min_q_all_actions = torch.min(q1_all_actions, q2_all_actions)

                    # Calculate Actor loss: J_pi = E_{s~D} [ sum_a [ pi(a|s) * (alpha * log pi(a|s) - Q_min(s,a)) ] ]
                    # We want to maximize this, so minimize its negative.
                    actor_loss_term = current_probs * (alpha.detach() * current_log_probs - min_q_all_actions.detach())
                    if lmbda != 0:
                        loss_actor = (torch.sum(actor_loss_term, dim=-1)).mean() + policy.regularization()
                    else:
                        loss_actor = (torch.sum(actor_loss_term, dim=-1)).mean()

                    # Optimize Actor
                    actor_optim.zero_grad()
                    loss_actor.backward()
                    actor_optim.step()

                    # Loss: E[log_alpha * (entropy - target_entropy)]
                    # We detach the entropy difference because we only optimize log_alpha here.
                    #    log_pi = torch.log(torch.tensor(self.pi[0:self.index], requires_grad=False))
                    #loss = torch.sum(torch.tensor(self.pi[0:self.index], requires_grad=False) * (-self.log_alpha * (log_pi + self.H)))
                
                    loss_alpha = ((current_probs.detach() * (-log_alpha) * (current_log_probs + target_entropy).detach()).sum(axis=1)).mean()

                    # Optimize Alpha
                    alpha_optim.zero_grad()
                    loss_alpha.backward()
                    alpha_optim.step()
                    #with torch.no_grad():
                    #    log_alpha.clamp_(min=-11.4)

                    # Get current alpha for logging AFTER optimizer step potentially changed log_alpha
                    alpha = log_alpha.exp().item()

                    # --- Update Target Networks ---
                    soft_update(target_q_net1, q_net1, tau)
                    soft_update(target_q_net2, q_net2, tau)

                #print(step)

                    # --- Logging (use manually calculated losses) ---
                if step % 100 == 0: # Log roughly every 1000 frames
                    save_path = f"Sparsity_Checkpoints\\chkpt_{int(lmbda*10000)}_{step}.pth"

                    # Ensure log_alpha value is current before saving optimizer state
                    # (though it should be if optimizer step happened)
                    current_log_alpha = log_alpha.detach().clone()

                    torch.save({
                        'actor_state_dict': policy.state_dict(),
                        'critic1_state_dict': q_net1.state_dict(),
                        'critic2_state_dict': q_net2.state_dict(),
                        'target1_state_dict': target_q_net1.state_dict(),
                        'target2_state_dict': target_q_net2.state_dict(),
                        'actor_optim_state_dict': actor_optim.state_dict(),
                        'critic1_optim_state_dict': critic1_optim.state_dict(),
                        'critic2_optim_state_dict': critic2_optim.state_dict(),
                        'alpha_optim_state_dict': alpha_optim.state_dict(),
                        # Explicitly save log_alpha value as well for robustness
                        'log_alpha': current_log_alpha
                    }, save_path)

                    print(f"Step={step:07d}  Actor Loss={loss_actor.item():.3f}"
                        f"  Q Loss={loss_q.item():.3f} ({loss_q1.item():.3f}, {loss_q2.item():.3f})"
                        f"  Alpha Loss={loss_alpha}  Alpha={alpha}"
                        f"  Buffer Size={len(replay_buffer)}")

        print("Training finished.")


if __name__ == "__main__":
    # this is critical on Windows when using spawn()
    # optionally explicitly set start method
    # Using spawn is generally safer for multiprocessing with CUDA
    main()