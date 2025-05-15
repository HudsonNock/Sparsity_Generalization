import torch
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F
from torchrl.envs import GymEnv, TransformedEnv, ParallelEnv
from torchrl.modules import ProbabilisticActor
from torchrl.objectives.sac import DiscreteSACLoss
from torchrl.data import Categorical
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from tensordict.nn import TensorDictModule
from torchrl.objectives import ValueEstimators
import torch.nn.functional as F
from tensordict import TensorDict
    
# helper to build a fresh CNN encoder
def make_cnn_encoder():
    return nn.Sequential(
        nn.Conv2d(1, 32, 4, stride=2), nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2), nn.ReLU(),
        nn.Conv2d(32, 16, 4, stride=1), nn.ReLU(),
        nn.Flatten(),
    )

def build_agents_and_buffer(obs_spec, act_spec, device):
    # Compute flattened feature size automatically
    with torch.no_grad():
        dummy = torch.zeros((1, *obs_spec.shape), device=device)
        dummy = torch.mean(dummy, dim=-1)[None, :, :, :] #[1,1,64,64]
        temp_encoder = make_cnn_encoder().to('cuda')
        n_flat = temp_encoder(dummy).shape[-1]

    # 2. Define actor network
    actor_encoder = make_cnn_encoder()
    actor_net = nn.Sequential(
        actor_encoder,
        nn.Linear(n_flat, 512), nn.ReLU(),
        nn.Linear(512, act_spec.shape[-1]),
    ).to(device)

    actor_td = TensorDictModule(
        module=actor_net,
        in_keys=["pixels"],
        out_keys=["logits"],
    )

    policy = ProbabilisticActor(
        module=actor_td,
        in_keys=["logits"],
        distribution_class=torch.distributions.Categorical,
        spec=act_spec,
        out_keys=["action"],
    ).to(device)

    class CriticNet(nn.Module):
        def __init__(self, encoder: nn.Module, n_flat: int, n_actions: int):
            super().__init__()
            self.encoder = encoder
            self.lin1    = nn.Linear(n_flat + n_actions, 512)
            self.lin2    = nn.Linear(512, 1)

        def forward(self, pixels: torch.Tensor, action: torch.Tensor):
            # encode pixels
            feats = self.encoder(pixels)                             # [B, n_flat]
            # concat one-hot
            x = torch.cat([feats, action], dim=-1)            # [B, n_flat + n_actions]
            x = F.relu(self.lin1(x))
            return self.lin2(x)                                      # [B,1]

    # then in your build function:
    critic_encoder = make_cnn_encoder()
    critic_net = CriticNet(critic_encoder, n_flat, act_spec.shape[-1]).to(device)

    qvalue_td = TensorDictModule(
        module=critic_net,
        in_keys=["pixels", "action"],
        out_keys=["state_action_value"],
    )

    # 4. Build the SAC loss
    sac_loss = DiscreteSACLoss(
        actor_network=policy,
        qvalue_network=qvalue_td,
        action_space=act_spec,
        num_actions=act_spec.shape[-1],
        num_qvalue_nets=2,
        alpha_init=0.2,
    ).to(device)
    #sac_loss.make_value_estimator(ValueEstimators.TDLambda, gamma=0.98, lmbda=0.3)
    sac_loss.make_value_estimator(ValueEstimators.TD0, gamma=0.98)

    # 5. Create a replay buffer
    buffer_size = 50_000
    batch_size  = 64
    storage = LazyTensorStorage(
        max_size=buffer_size,
    )
    replay_buffer = TensorDictReplayBuffer(
        storage=storage,
        sampler=RandomSampler(),
        batch_size=batch_size,
    )

    # 6. Set up optimizers
    optim = torch.optim.Adam(
        [
            {"params": policy.parameters(),         "lr": 3e-4},
            {"params": qvalue_td.parameters(),      "lr": 3e-4},
            {"params": sac_loss._alpha,             "lr": 1e-4},  # train temperature
        ]
    )

    return policy, qvalue_td, sac_loss, replay_buffer, optim


def main():
    num_envs = 8
    env_name = "procgen:procgen-caveflyer"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = ParallelEnv(
        num_workers=num_envs,
        create_env_fn=lambda: GymEnv(env_name, frame_skip=1),
        device=device
    )
    #env = GymEnv(env_name, frame_skip=1, device=device)
    #env = TransformedEnv(raw_env)

    # pull specs
    obs_spec    = env.specs[0]["output_spec"]["full_observation_spec"]["pixels"]
    act_spec    = env.specs[0]["input_spec"]["full_action_spec"]["action"]
    rew_spec    = env.specs[0]["output_spec"]["full_reward_spec"]["reward"]
    done_spec   = env.specs[0]["output_spec"]["full_done_spec"]["done"]

    policy, qvalue, sac_loss, replay_buffer, optim = build_agents_and_buffer(
        obs_spec, act_spec, device
    )

    # 7. Training loop
    td = env.reset()
    td["pixels"] = torch.mean(td["pixels"], dim=-1, dtype=torch.float32)[:, None, :, :]
    td_old = None

    num_steps       = 200_000
    update_every    = 5
    collect_steps   = 10
    batch_size      = 64

    for step in range(1, num_steps + 1):
        # 7a. collect a batch of transitions
        for j in range(collect_steps):
            # run policy
            td = policy(td)
            oh = F.one_hot(td["action"], num_classes=act_spec.shape[-1]).to(td.device)
            td.set("action", oh)

            # step env
            td = env.step(td)
            td["next"]["pixels"] = torch.mean(td["next"]["pixels"], dim=-1, dtype=torch.float32)[:, None, :, :]
            # package a transition tensordict
            if j > 0:
                trans = TensorDict({
                    "pixels":      td_old["pixels"],
                    "action":      td_old["action"],
                    "logits":      td_old["logits"],
                    "next": TensorDict({
                        "reward": td_old["next"]["reward"],
                        "done": td_old["next"]["done"],
                        "pixels": td_old["next"]["pixels"]
                    }, batch_size=[num_envs], device='cpu'),
                }, batch_size=[num_envs], device='cpu')

                for single_td in trans.unbind(0):   # this gives you 8 TensorDicts of batch_size [1]
                    replay_buffer.add(single_td)

            td_old = td
            td = td["next"]

        # 7b. update every few env steps
        if step % update_every == 0 and len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size=batch_size).to(device)  # tensordict of size [batch_size, ...]

            #print(batch["action"].shape)
            # compute all SAC losses
            #loss_dict = sac_loss(
            #    observation=batch["pixels"],
            #    action=batch["action"],
            #    next_done=batch["next"]["done"],
            #    next_terminated=batch["next"]["done"]
            #    next_observation=batch["next"]["pixels"],
            #    next_reward=batch["next"]["reward"]
            #)
            loss_dict = sac_loss(batch)

            # sum actor + critic + alpha losses
            total_loss = (
                loss_dict["loss_actor"].mean()
            + loss_dict["loss_qvalue"].mean()
            + loss_dict["loss_alpha"].mean()
            )

            optim.zero_grad()
            total_loss.backward()
            optim.step()

            print("A")

        # optional logging
        if step % 1000 == 0:
            print(f"step={step:07d}  actor={loss_dict['loss_actor'].item():.3f}"
                f"  q={loss_dict['loss_qvalue'].item():.3f}"
                f"  alpha={loss_dict['loss_alpha'].item():.3f}")


if __name__ == "__main__":
    # this is critical on Windows when using spawn()
    mp.freeze_support()
    # optionally explicitly set start method
    mp.set_start_method("spawn", force=True)
    # then kick off main
    main()
