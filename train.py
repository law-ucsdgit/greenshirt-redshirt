from __future__ import annotations

import os
from collections import deque

import numpy as np
import torch

from connect4.config import TrainingConfig
from connect4.game import rlEnvironment
from connect4.model import DQN, ReplayMemory, optimize_model, select_action

cfg = TrainingConfig()
os.makedirs(cfg.save_dir, exist_ok=True)

env = rlEnvironment()

policy_net1 = DQN(42, 7)
target_net1 = DQN(42, 7)
policy_net2 = DQN(42, 7)
target_net2 = DQN(42, 7)

target_net1.load_state_dict(policy_net1.state_dict())
target_net2.load_state_dict(policy_net2.state_dict())

optimizer1 = torch.optim.AdamW(policy_net1.parameters(), lr=cfg.lr, amsgrad=True)
optimizer2 = torch.optim.AdamW(policy_net2.parameters(), lr=cfg.lr, amsgrad=True)

memory1 = ReplayMemory(cfg.memory_size)
memory2 = ReplayMemory(cfg.memory_size)

p1_wins = deque(maxlen=100)
p2_wins = deque(maxlen=100)
steps_done = 0


def save_checkpoint(model, optimizer, episode, filename):
    path = os.path.join(cfg.save_dir, filename)
    checkpoint = {
        "episode": episode,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


for i_episode in range(cfg.episodes):
    env.reset()
    state = torch.tensor(env.reset(), dtype=torch.float32).view(1, -1)
    pstate = None
    paction = None
    preward = None
    pnext_state = None

    while True:
        current_player = env.steps % 2
        if current_player == 0:
            policy_net = policy_net1
            target_net = target_net1
            optimizer = optimizer1
            memory = memory1
        else:
            policy_net = policy_net2
            target_net = target_net2
            optimizer = optimizer2
            memory = memory2

        action = select_action(
            state,
            policy_net,
            env.action_space,
            current_player,
            steps_done,
            cfg.eps_start,
            cfg.eps_end,
            cfg.eps_decay,
        )
        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        next_state = None if terminated else torch.tensor(observation, dtype=torch.float32).view(1, -1)

        if pstate is not None:
            memory.push(pstate, paction, pnext_state, torch.tensor([preward], dtype=torch.float32))

        pstate = state
        paction = action.view(1, 1)
        preward = float(reward)
        pnext_state = next_state
        state = next_state if next_state is not None else state

        optimize_model(memory, policy_net, target_net, optimizer, cfg.batch_size, cfg.gamma)

        target_state_dict = target_net.state_dict()
        policy_state_dict = policy_net.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key] * cfg.tau + target_state_dict[key] * (1 - cfg.tau)
        target_net.load_state_dict(target_state_dict)

        if done:
            if reward >= 100:
                winner = 0
            elif reward <= -100:
                winner = 1
            else:
                winner = 1 - current_player

            p1_wins.append(1 if winner == 0 else 0)
            p2_wins.append(1 if winner == 1 else 0)
            break

    if (i_episode + 1) % cfg.checkpoint_interval == 0:
        save_checkpoint(policy_net1, optimizer1, i_episode, f"policy_net1_ep{i_episode + 1}.pt")
        save_checkpoint(policy_net2, optimizer2, i_episode, f"policy_net2_ep{i_episode + 1}.pt")

print("Training complete.")
print(f"Player 1 win rate: {np.mean(p1_wins) if len(p1_wins) else 0.0}")
print(f"Player 2 win rate: {np.mean(p2_wins) if len(p2_wins) else 0.0}")
