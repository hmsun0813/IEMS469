"""
Assignment 2 — Deep Q-Network (DQN)

Implements DQN for:
  (a) CartPole-v1 (gamma = 0.95)
  (b) MsPacman-v5 (gamma = 0.99) 

Features required:
  • Neural network approximator
  • Replay buffer
  • Target network

Extras included for performance: Huber loss option, LR scheduler (StepLR), epsilon scheduler.

Usage examples (single-GPU selection):
  # Train CartPole for 1000 episodes on GPU 0
  CUDA_VISIBLE_DEVICES=0 python dqn_hw2.py --env cartpole --episodes 1000 --device cuda:0

  # Train MsPacman for 5000 episodes on GPU 1 with Huber loss and a StepLR scheduler
  CUDA_VISIBLE_DEVICES=1 python dqn_hw2.py --env mspacman --episodes 5000 --huber --lr-step 20000 --lr-gamma 0.5 --device cuda:1

This file saves plots and checkpoints under ./outputs and ./ckpts respectively.

Ensure the correct gym package. For Atari MsPacman, modern setups use Gymnasium with ALE
"""
from __future__ import annotations
import os
import math
import random
import argparse
from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

# --------------------------- Utilities ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(env_name: str):
    """
    Create environment by name, supporting both Gym and Gymnasium ids.
    """
    try:
        import gymnasium as gym
        import ale_py
    except Exception:
        import gym as gym  # fallback for older installs

    # Accept aliases
    if env_name.lower() in ["mspacman", "ms_pacman", "mspacman-v5", "ale/mspacman-v5"]:
        candidates = ["ALE/MsPacman-v5", "MsPacman-v5"]
    elif env_name.lower() in ["cartpole", "cartpole-v1"]:
        candidates = ["CartPole-v1"]
    else:
        candidates = [env_name]

    last_err = None
    for cid in candidates:
        try:
            env = gym.make(cid, render_mode=None)
            return env
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not create env from candidates {candidates}: {last_err}")


# MsPacman preprocessing as specified in the assignment
MSPACMAN_COLOR = 210 + 164 + 74

def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    img = obs[1:176:2, ::2]
    img = img.sum(axis=2)
    img[img == MSPACMAN_COLOR] = 0
    img = (img // 3 - 128).astype(np.int8)
    return img.reshape(88, 80, 1)

def obs_to_state(o, env_kind: str):
    if env_kind == "mspacman":
        proc = preprocess_observation(o).astype(np.float32) / 128.0
        proc = np.transpose(proc, (2, 0, 1))  # (1,88,80)
        return proc
    else:
        return o.astype(np.float32)


# --------------------------- Replay Buffer ---------------------------

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: Optional[np.ndarray]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], dtype=np.float32):
        self.capacity = capacity
        self.pos = 0
        self.full = False
        self.states = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def push(self, s, a, r, ns, d):
        i = self.pos
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        if ns is None:
            self.next_states[i] = 0
        else:
            self.next_states[i] = ns
        self.dones[i] = d
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.pos

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self), size=batch_size)
        return (
            torch.from_numpy(self.states[idx]).float(),
            torch.from_numpy(self.actions[idx]).long(),
            torch.from_numpy(self.rewards[idx]).float(),
            torch.from_numpy(self.next_states[idx]).float(),
            torch.from_numpy(self.dones[idx]).float(),
        )


# --------------------------- Q-Networks ---------------------------

class MLPQ(nn.Module):
    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class CNNDQN(nn.Module):
    def __init__(self, in_ch: int, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 88, 80)
            n_flat = self.features(dummy).view(1, -1).size(1)
        self.head = nn.Sequential(
            nn.Linear(n_flat, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


# --------------------------- Schedulers ---------------------------

def linear_epsilon_scheduler(frame_idx: int, eps_start: float, eps_end: float, eps_decay_frames: int) -> float:
    if eps_decay_frames <= 0:
        return eps_end
    t = min(1.0, frame_idx / float(eps_decay_frames))
    return eps_start + t * (eps_end - eps_start)


# --------------------------- Training / Evaluation ---------------------------

def compute_td_loss(batch, online_net, target_net, gamma, device, huber=False):
    states, actions, rewards, next_states, dones = [x.to(device) for x in batch]

    # Q(s,a)
    q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # max_a' Q_target(s', a') with zero for terminal
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target = rewards + gamma * next_q * (1.0 - dones)

    if huber:
        loss = nn.SmoothL1Loss()(q_values, target)
    else:
        loss = nn.MSELoss()(q_values, target)
    return loss, q_values.detach(), target.detach()


def to_tensor_obs(obs, env_kind: str, device):
    """Wrap a *state* array into a torch tensor on device.
    Expectation: obs_to_state() has already produced the correct shape.
    - MsPacman: (1, 88, 80) CHW float32 scaled ~[-1,1]
    - CartPole: (obs_dim,) float32
    """
    if env_kind == "mspacman":
        arr = obs.astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0).to(device)
    else:
        return torch.from_numpy(obs).float().unsqueeze(0).to(device)


def get_obs_shape(env, env_kind: str) -> Tuple[int, ...]:
    if env_kind == "mspacman":
        return (1, 88, 80)
    else:
        return (env.observation_space.shape[0],)


def get_n_actions(env) -> int:
    try:
        return env.action_space.n
    except Exception:
        raise ValueError("Only discrete action spaces are supported in this implementation.")


# Moving average

def moving_average(x: List[float], w: int = 100) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    w = max(1, w)
    cumsum = np.cumsum(np.insert(np.array(x, dtype=float), 0, 0))
    ma = (cumsum[w:] - cumsum[:-w]) / float(w)
    pad = np.full((w - 1,), np.nan) # Left-pad to align lengths
    return np.concatenate([pad, ma])


# Plot helpers

def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("ckpts", exist_ok=True)


def plot_training_curves(max_q_history, rewards_history, env_tag: str):
    ensure_dirs()
    # (i) Max Q vs episodes
    plt.figure()
    plt.plot(max_q_history)
    plt.xlabel("Episode")
    plt.ylabel("Max Q (per episode)")
    plt.title(f"Max Q vs Episodes — {env_tag}")
    plt.tight_layout()
    plt.savefig(f"outputs/{env_tag}_maxQ_vs_episodes.png")
    plt.close()

    # (ii) Rewards vs episodes with MA(100)
    ma100 = moving_average(rewards_history, 100)
    plt.figure()
    plt.plot(rewards_history, label="Episode Reward")
    plt.plot(ma100, label="Moving Avg (100)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.title(f"Episode Rewards — {env_tag}")
    plt.tight_layout()
    plt.savefig(f"outputs/{env_tag}_rewards_vs_episodes.png")
    plt.close()


def plot_eval_hist(all_rewards: List[float], env_tag: str):
    ensure_dirs()
    plt.figure()
    plt.hist(all_rewards, bins=30)
    plt.xlabel("Episode Reward")
    plt.ylabel("Count")
    plt.title(f"Evaluation Reward Distribution (n={len(all_rewards)}) — {env_tag}")
    plt.tight_layout()
    plt.savefig(f"outputs/{env_tag}_eval_hist.png")
    plt.close()


# --------------------------- Main Runner ---------------------------

def train_and_eval(args):
    set_seed(args.seed)

    # Single-GPU selection: respect CUDA_VISIBLE_DEVICES and explicit --device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    env_kind = "mspacman" if args.env.lower().startswith("mspac") else "cartpole"
    env = make_env(args.env)
    n_actions = get_n_actions(env)

    # Discount per assignment
    gamma = args.gamma
    if args.gamma is None:
        gamma = 0.99 if env_kind == "mspacman" else 0.95

    # Observation shape and models
    obs_shape = get_obs_shape(env, env_kind)

    if env_kind == "mspacman":
        online_net = CNNDQN(in_ch=obs_shape[0], n_actions=n_actions)
        target_net = CNNDQN(in_ch=obs_shape[0], n_actions=n_actions)
    else:
        in_dim = obs_shape[0]
        online_net = MLPQ(in_dim, n_actions)
        target_net = MLPQ(in_dim, n_actions)

    online_net.to(device)
    target_net.to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_step > 0 and args.lr_gamma < 1.0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # Replay buffer
    rb_dtype = np.float32
    if env_kind == "mspacman":
        rb_dtype = np.float32  # we store float32 scaled frames
    buffer = ReplayBuffer(args.replay_size, obs_shape, dtype=rb_dtype)

    # Logging
    writer = SummaryWriter(log_dir=args.tb_logdir) if args.tb_logdir else None


    # Warm-up: collect random steps to fill buffer minimally
    try:
        import gymnasium as gym
        import ale_py
    except Exception:
        import gym as gym

    obs, _ = env.reset(seed=args.seed) if hasattr(env, "reset") and len(env.reset.__code__.co_varnames) > 1 else (env.reset(), {})

    state = obs_to_state(obs, env_kind)

    frame_idx = 0
    start_ep = 1
    episode_rewards: List[float] = []
    max_q_per_ep: List[float] = []

    # ---------------- Resume from checkpoint (optional) ----------------
    if args.load is not None and os.path.exists(args.load):
        print(f"Resuming training from {args.load}")
        ckpt = torch.load(args.load, map_location=device)
        if isinstance(ckpt, dict) and ("model" in ckpt or "optimizer" in ckpt):
            if "model" in ckpt:
                online_net.load_state_dict(ckpt["model"])
                target_net.load_state_dict(online_net.state_dict())
            if "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    for g in optimizer.param_groups:
                        g["lr"] = args.lr
                except Exception as e:
                    print(f"[warn] couldn't load optimizer state: {e}")
            frame_idx = int(ckpt.get("frame_idx", 0))
            # Restore full histories so plots span the whole trajectory
            episode_rewards = list(ckpt.get("rewards", []))
            max_q_per_ep   = list(ckpt.get("max_q", []))
            start_ep = int(ckpt.get("episode", 0)) + 1
        else:
            # If it's a raw state_dict (model only)
            online_net.load_state_dict(ckpt)
            target_net.load_state_dict(online_net.state_dict())
        print(f"Resume state: start_ep={start_ep}, frame_idx={frame_idx}, hist_len={len(episode_rewards)}")


    # Training loop over episodes
    pbar = tqdm(range(start_ep, start_ep + args.episodes), desc=f"Training {args.env}")
    for ep in pbar:
        done = False
        ep_reward = 0.0
        ep_max_q = -float('inf')

        # Reset env per episode
        obs, _ = env.reset()
        state = obs_to_state(obs, env_kind)

        while not done:
            frame_idx += 1
            # Epsilon-greedy
            eps = linear_epsilon_scheduler(
                frame_idx,
                args.eps_start,
                args.eps_end,
                args.eps_decay_frames,
            )

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    qvals = online_net(to_tensor_obs(state, env_kind, device)).cpu().numpy()[0]
                    action = int(np.argmax(qvals))
                    ep_max_q = max(ep_max_q, float(np.max(qvals)))

            # Step environment
            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_out

            next_state = obs_to_state(next_obs, env_kind) if not done else None
            buffer.push(state, action, reward, next_state if next_state is not None else state, done)
            state = next_state if next_state is not None else state
            ep_reward += float(reward)

            # Optimize
            if len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                loss, q_values, target = compute_td_loss(
                    batch, online_net, target_net, gamma, device, huber=args.huber
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), args.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            # Target network update
            if frame_idx % args.target_update == 0:
                target_net.load_state_dict(online_net.state_dict())

        # Episode end logging
        episode_rewards.append(ep_reward)
        max_q_per_ep.append(ep_max_q if ep_max_q != -float('inf') else float('nan'))

        if writer:
            writer.add_scalar("train/episode_reward", ep_reward, ep)
            writer.add_scalar("train/max_q", max_q_per_ep[-1], ep)
            writer.add_scalar("train/epsilon", eps, ep)

        pbar.set_postfix({
            "ep_reward": f"{ep_reward:.1f}",
            "eps": f"{eps:.3f}",
            "frames": frame_idx,
        })

        # Add inline plotting for training progress every 500 episodes
        if ep % 500 == 0 and ep > 0:
            ensure_dirs()
            tag = "mspacman" if env_kind == "mspacman" else "cartpole"
            plt.figure()
            plt.plot(max_q_per_ep)
            plt.xlabel("Episode")
            plt.ylabel("Max Q")
            plt.title(f"Max Q up to episode {ep}")
            plt.tight_layout()
            plt.savefig(f"outputs/{tag}_maxQ_partial_ep{ep}.png")
            plt.close()

            ma100 = moving_average(episode_rewards, 100)
            plt.figure()
            plt.plot(episode_rewards, label="Episode Reward")
            plt.plot(ma100, label="Moving Avg (100)")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"Rewards up to episode {ep}")
            plt.tight_layout()
            plt.savefig(f"outputs/{tag}_rewards_partial_ep{ep}.png")
            plt.close()


        # Save periodic checkpoints
        if args.checkpoint_every > 0 and ep % args.checkpoint_every == 0:
            ensure_dirs()
            tag = "mspacman" if env_kind == "mspacman" else "cartpole"
            torch.save({
                "model": online_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": ep,
                "frame_idx": frame_idx,
                "rewards": episode_rewards,
                "max_q": max_q_per_ep,
            }, f"ckpts/{tag}_ep{ep}.pt")

    # Final save
    ensure_dirs()
    tag = "mspacman" if env_kind == "mspacman" else "cartpole"
    latest_path = f"ckpts/{tag}_latest.pt"
    torch.save({
        "model": online_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": (start_ep + args.episodes - 1),
        "frame_idx": frame_idx,
        "rewards": episode_rewards,
        "max_q": max_q_per_ep,
    }, latest_path)

    # Plots (i) and (ii)
    plot_training_curves(max_q_per_ep, episode_rewards, tag)

    # Evaluation (iii)
    all_rewards = evaluate_policy(env, online_net, n_episodes=args.eval_episodes, env_kind=env_kind, device=device)
    plot_eval_hist(all_rewards, tag)
    mean_r = float(np.mean(all_rewards)) if len(all_rewards) else float('nan')
    std_r = float(np.std(all_rewards)) if len(all_rewards) else float('nan')

    print(f"Evaluation over {len(all_rewards)} episodes — mean: {mean_r:.2f}, std: {std_r:.2f}")
    print(f"Saved: outputs/{tag}_maxQ_vs_episodes.png, outputs/{tag}_rewards_vs_episodes.png, outputs/{tag}_eval_hist.png")
    print(f"Checkpoint: {latest_path}")


@torch.no_grad()
def evaluate_policy(env, net, n_episodes: int, env_kind: str, device) -> List[float]:
    net.eval()
    rewards = []
    pbar = tqdm(range(n_episodes), desc="Evaluating", leave=False)

    for _ in pbar:
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        state = None
        while not done:
            if state is None:
                state = obs_to_state(obs, env_kind)
            state_t = to_tensor_obs(state, env_kind, device)
            qvals = net(state_t).cpu().numpy()[0]
            action = int(np.argmax(qvals))

            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_out
            ep_r += float(reward)

            state = obs_to_state(next_obs, env_kind) if not done else state
            obs = next_obs
        rewards.append(ep_r)
        pbar.set_postfix({"last_ep_r": f"{ep_r:.1f}"})
    net.train()
    return rewards


# --------------------------- CLI ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DQN for CartPole and MsPacman with single-GPU and tqdm.")
    parser.add_argument("--env", type=str, default="cartpole", help="cartpole | mspacman | exact env id")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0", help="e.g., cuda:0 or cpu")

    # DQN hyperparams
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-step", type=int, default=0, help="LR StepLR step_size in optimizer steps; 0=disabled")
    parser.add_argument("--lr-gamma", type=float, default=1.0, help="LR StepLR gamma; <1 enables decay")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--target-update", type=int, default=1000, help="steps between target net syncs")
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=None, help="Override discount; default per assignment")

    # Epsilon schedule
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-frames", type=int, default=200_000)

    # Loss options
    parser.add_argument("--huber", action="store_true", help="Use Huber (SmoothL1) loss instead of MSE")

    # Logging / checkpoints
    parser.add_argument("--tb-logdir", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=100)

    # Eval-only mode
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--load", type=str, default=None)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.eval_only:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        env_kind = "mspacman" if args.env.lower().startswith("mspac") else "cartpole"
        env = make_env(args.env)
        n_actions = get_n_actions(env)
        obs_shape = get_obs_shape(env, env_kind)
        if env_kind == "mspacman":
            net = CNNDQN(in_ch=obs_shape[0], n_actions=n_actions).to(device)
        else:
            net = MLPQ(obs_shape[0], n_actions=n_actions).to(device)

        if args.load is None:
            raise ValueError("--eval-only requires --load checkpoint path")

        sd = torch.load(args.load, map_location=device)

        # Try to load model AND history from a bundle checkpoint; fall back to raw state_dict.
        rewards_hist = []
        maxq_hist = []
        if isinstance(sd, dict) and "model" in sd:
            net.load_state_dict(sd["model"])
            rewards_hist = sd.get("rewards", []) or []
            maxq_hist   = sd.get("max_q", []) or []
        else:
            net.load_state_dict(sd)

        # Evaluate current policy and plot eval histogram
        rewards = evaluate_policy(env, net, n_episodes=args.eval_episodes, env_kind=env_kind, device=device)
        tag = "mspacman" if env_kind == "mspacman" else "cartpole"
        plot_eval_hist(rewards, tag)

        # If the checkpoint had history, also (re)generate the two training plots from it
        if len(rewards_hist) and len(maxq_hist):
            plot_training_curves(maxq_hist, rewards_hist, tag)
            print(f"Eval-only: saved training curves from checkpoint history -> "
                f"outputs/{tag}_maxQ_vs_episodes.png, outputs/{tag}_rewards_vs_episodes.png")
        else:
            print("Eval-only: checkpoint had no 'rewards'/'max_q' history; only eval histogram was created.")

        mean_r, std_r = float(np.mean(rewards)), float(np.std(rewards))
        print(f"Eval-only — mean: {mean_r:.2f}, std: {std_r:.2f}")
        return


    # Otherwise, train then evaluate
    train_and_eval(args)


if __name__ == "__main__":
    main()
