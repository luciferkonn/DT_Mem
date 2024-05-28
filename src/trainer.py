import os
from typing import Callable, Optional, Union
import numpy as np
import torch
import time
import wandb
from tqdm import tqdm
from jax.tree_util import tree_map
from torch.utils.data.dataloader import DataLoader
from src.minlora.utils import get_lora_state_dict
from src.env_utils import ATARI_RETURN_RANGE
from src.utils import accuracy, autoregressive_generate, cross_entropy, decode_return, encode_return, encode_reward, sample_from_logits


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset_list,
        train_game_list,
        args,
        eval_env_list,
        eval_game_name,
        optimizer: Union[torch.optim.Optimizer, Callable],
        run_dir: str,
        grad_norm_clip: float,
        single_return_token: bool = True,
        num_steps_per_iter: int = 2500,
        log_interval: int = None,
        eval_log_interval: int = 100,
        use_wandb: bool = False,
        training_samples: int = 1000,
        eval_freq: int = 10,
        n_gpus: bool = False,
    ) -> None:
        self.model = model
        self.train_dataset_list = train_dataset_list
        self.train_game_list = train_game_list
        self.eval_envs = eval_env_list
        self.eval_game_name = eval_game_name
        self.args = args
        self.optimizer = None
        self.device = args.device
        self.num_steps_per_iter = num_steps_per_iter
        self.log_interval = log_interval
        self.eval_log_interval = eval_log_interval
        self.use_wandb = use_wandb
        self.grad_norm_clip = grad_norm_clip
        self.save_freq = args.save_freq
        self.max_epochs = args.max_epochs
        self.run_dir = run_dir
        self.model.to(device=self.device)
        self.n_gpus = n_gpus
        self.return_range = ATARI_RETURN_RANGE
        self.single_return_token = single_return_token
        self.training_samples = training_samples
        self.eval_freq = eval_freq
        self.warmup_steps = args.warmup_steps

    def train(self, current_epoch, optimizer, apply_lora=False):
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda steps: min((steps+1)/self.warmup_steps, 1))
        if apply_lora:
            tf_file_loc = os.path.join(self.run_dir, f'lora_model.pt')
        else:
            tf_file_loc = os.path.join(
                self.run_dir, f'tf_model.pt')
        for epoch in range(current_epoch, self.args.max_epochs):
            # train model
            logs = self.run_epoch(iter_num=epoch)
            if epoch % self.save_freq == 0 or epoch == (self.max_epochs - 1):
                print("========================")
                print(f"The model is saved to {tf_file_loc}")
                if apply_lora:
                    lora_state_dict = get_lora_state_dict(self.model)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': lora_state_dict,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': logs['train_loss']
                    }, tf_file_loc)
                else:

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': logs['train_loss']
                    }, tf_file_loc)
            # evaluate model
            # if self.args.eval:
            if epoch % self.eval_freq == 0:
                rew_sum = self.evaluation_rollout(
                    eval_envs_list=self.eval_envs, num_steps=self.args.eval_steps,
                    eval_log_interval=self.eval_log_interval, device=self.device)

    def evaluate(self):
        rew_sum = self.evaluation_rollout(
            eval_envs_list=self.eval_envs, num_steps=self.args.eval_steps,
            eval_log_interval=self.eval_log_interval, device=self.device)

    def load_model(self, model_path, apply_lora=False):
        ckpt = torch.load(model_path)
        # if not apply_lora:
            # self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt['epoch']
        loss = ckpt['loss']
        return epoch, loss

    def run_epoch(
        self,
        iter_num: int,
    ):
        # Prepare some log infos
        # trainer_loss = []
        logs = dict()
        train_start = time.time()
        self.model.train()
        for data, game_name in zip(self.train_dataset_list, self.train_game_list):
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=self.args.batch_size,
                                num_workers=self.args.num_workers)

            pbar = tqdm(enumerate(loader), total=self.training_samples)
            # pbar = tqdm(enumerate(loader), total=len(loader))
            n_samples = 0
            for t, (obs, rtg, actions, rewards) in pbar:
                obs = obs.to(self.device)
                rtg = rtg.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                inputs = {'observations': obs,
                          'returns-to-go': rtg,
                          'actions': actions,
                          'rewards': rewards}
                with torch.set_grad_enabled(True):
                    result_dict = self.model(inputs=inputs)
                    train_loss = result_dict['loss']
                self.optimizer.zero_grad(set_to_none=True)
                if self.n_gpus:
                    train_loss.mean().backward()
                else:
                    train_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_norm_clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.log_interval and t % self.log_interval == 0:
                    if self.n_gpus:
                        train_loss = train_loss.mean().detach().cpu().item()
                        acc = result_dict['accuracy'].mean(
                        ).detach().cpu().item()*100
                    else:
                        train_loss = train_loss.detach().cpu().item()
                        acc = result_dict['accuracy'].detach().cpu().item()*100
                    pbar.set_description(
                        f"game {game_name} epoch {iter_num} steps: {t}: train loss {train_loss:.5f} accuracy {acc:.3f}%.")
                    if self.use_wandb:
                        wandb.log({f"train/episode_loss/{game_name}": train_loss,
                                   f"train/accuracy/{game_name}": acc,
                                   f'train/epoch/{game_name}': (iter_num)*self.num_steps_per_iter+t})
                if n_samples >= self.training_samples:
                    break
                n_samples += 1
        training_time = time.time() - train_start
        logs['time/training'] = training_time
        logs['train_loss'] = train_loss
        return logs

    def evaluation_rollout(
        self,
        eval_envs_list,
        num_steps=2500,
        eval_log_interval=None,
        device='cpu',
    ):
        self.model.eval()
        """Roll out a batch of environments under a given policy function."""
        # observations are dictionaries. Merge into single dictionary with batched
        # observations.
        for envs, game_name in zip(eval_envs_list, self.eval_game_name):
            obs_list = [env.reset() for env in envs]
            num_batch = len(envs)
            obs = tree_map(lambda *arr: torch.from_numpy(np.stack(arr,
                                                                  axis=0)).to(device=device), *obs_list)
            obs['observations'] = obs['observations'].permute(0, 4, 1, 2, 3)
            # ret = np.zeros([num_batch, 8])
            done = np.zeros(num_batch, dtype=np.int32)
            rew_sum = np.zeros(num_batch, dtype=np.float32)

            # frames = []
            for t in range(num_steps):
                # Collect observations
                # frames.append(
                #     np.concatenate([o['observations'][-1, ...] for o in obs_list], axis=1))
                done_prev = done

                actions = get_action(
                    inputs=obs, model=self.model, return_range=self.return_range,
                    single_return_token=self.single_return_token, opt_weight=0, num_samples=128,
                    action_temperature=1.0, return_temperature=0.75,
                    action_top_percentile=50, return_top_percentile=None)

                # Collect step results and stack as a batch.
                step_results = [env.step(act.detach().cpu().numpy())
                                for env, act in zip(envs, actions)]
                # print("=======>Actions")
                # print(actions)
                obs_list = [result[0] for result in step_results]
                obs = tree_map(
                    lambda *arr: torch.from_numpy(np.stack(arr, axis=0)).to(device=device), *obs_list)
                obs['observations'] = obs['observations'].permute(
                    0, 4, 1, 2, 3)
                rew = np.stack([result[1] for result in step_results])
                done = np.stack([result[2] for result in step_results])
                # Advance state.
                done = np.logical_or(done, done_prev).astype(np.int32)
                rew = rew * (1 - done)
                rew_sum += rew
                if eval_log_interval and t % eval_log_interval == 0:
                    print('game: %s step: %d done: %s reward: %s' %
                          (game_name, t, done, rew_sum))
                # Don't continue if all environments are done.
                if np.all(done):
                    break
            print('game: %s step: %d done: %s reward: %s' %
                  (game_name, t, done, rew_sum))
            top3 = rew_sum[np.argsort(rew_sum)][-3:]
            print(f"mean {np.mean(rew_sum)}, top3 {np.mean(top3)}")
            if self.use_wandb:
                wandb.log({f"eval/step/{game_name}": t,
                           f"eval/rew_mean/{game_name}": np.mean(rew_sum)})
        return np.mean(rew_sum)


def get_action(
    inputs,
    model,
    return_range,
    single_return_token,
    opt_weight: Optional[float] = 0.0,
    num_samples: Optional[int] = 128,
    action_temperature: Optional[float] = 1.0,
    return_temperature: Optional[float] = 1.0,
    action_top_percentile: Optional[float] = None,
    return_top_percentile: Optional[float] = None,
):
    obs, act, rew = inputs['observations'], inputs['actions'], inputs['rewards']
    assert len(obs.shape) == 5
    assert len(act.shape) == 2
    act = act[:, -1].unsqueeze(1)
    inputs['actions'] = act
    inputs['rewards'] = rew[:, -1].unsqueeze(1)
    inputs['returns-to-go'] = torch.zeros_like(act)
    seq_len = obs.shape[1]
    timesteps = -1

    def ret_sample_fn(logits):
        assert len(logits.shape) == 2
        # Add optimality bias
        if opt_weight > 0.0:
            # Calculate log of P(optimality|return) = exp(return)/Z
            logits_opt = torch.linspace(0., 1., logits.shape[1])
            logits_opt = torch.repeat_interleave(
                logits_opt.unsqueeze(0), logits.shape[0], dim=0)
            # Sample from log[P(optimality=1|return)*P(return)]
            logits = logits + opt_weight * logits_opt
        logits = torch.repeat_interleave(
            logits.unsqueeze(0), num_samples, dim=0)
        ret_sample = sample_from_logits(
            logits, temperature=return_temperature, top_percentile=return_top_percentile)
        # pick the highest return sample
        ret_sample = torch.max(ret_sample)
        # ret_sample = torch.max(ret_sample, dim=0)
        # Convert return tokens into return values
        ret_sample = decode_return(ret_sample, return_range)
        return ret_sample

    with torch.no_grad():
        if single_return_token:
            ret_logits = model(inputs)['return_logits'][:, 0, :]
            ret_sample = ret_sample_fn(ret_logits)
            inputs['returns-to-go'][:, 0] = ret_sample
        else:
            # Auto-regressively regenerate all return tokens in a sequence
            def ret_logits_fn(ipts): return model(ipts)[
                'return_logits']
            ret_sample = autoregressive_generate(
                inputs, ret_logits_fn, 'returns-to-go', seq_len, ret_sample_fn)
            inputs['returns-to-go'] = ret_sample

        # Generate a sample from action logits
        act_logits = model(inputs)['action_logits'][:, timesteps, :]
        act_sample = sample_from_logits(
            act_logits, temperature=action_temperature,
            top_percentile=action_top_percentile)
    return act_sample
