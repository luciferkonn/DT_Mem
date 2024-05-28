import collections
import gym
import numpy as np
from src.env_utils import AtariEnvWrapper


class WrappedGymEnv:

    def __getattr__(self, name):
        """Wrappers forward non-overridden method calls to their wrapped env."""
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)


class SequenceEnvironmentWrapper(WrappedGymEnv):
    """Environment wrapper for supporting sequential model inference.
    """

    def __init__(self,
                 env,
                 num_stack_frames: int = 1):
        self._env = env
        self.num_stack_frames = num_stack_frames
        if self.is_goal_conditioned:
            # If env is goal-conditioned, we want to track goal history.
            self.goal_stack = collections.deque(
                [], maxlen=self.num_stack_frames)
        self.obs_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.act_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.rew_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.done_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.info_stack = collections.deque([], maxlen=self.num_stack_frames)

    @property
    def observation_space(self):
        """Constructs observation space."""
        parent_obs_space = self._env.observation_space
        act_space = self.action_space
        episode_history = {
            'observations': gym.spaces.Box(
                np.stack([parent_obs_space.low] *
                         self.num_stack_frames, axis=0),
                np.stack([parent_obs_space.high] *
                         self.num_stack_frames, axis=0),
                dtype=parent_obs_space.dtype),
            'actions': gym.spaces.Box(
                0, act_space.n, [self.num_stack_frames], dtype=act_space.dtype),
            'rewards': gym.spaces.Box(-np.inf, np.inf, [self.num_stack_frames])
        }
        if self.is_goal_conditioned:
            # pytype: disable=attribute-error
            goal_shape = np.shape(self._env.goal)
            episode_history['returns-to-go'] = gym.spaces.Box(
                -np.inf, np.inf, [self.num_stack_frames] + goal_shape)
        return gym.spaces.Dict(**episode_history)

    @property
    def is_goal_conditioned(self):
        return False

    def pad_current_episode(self, obs, n):
        # Prepad current episode with n steps.
        for _ in range(n):
            if self.is_goal_conditioned:
                # pytype: disable=attribute-error
                self.goal_stack.append(self._env.goal)
            self.obs_stack.append(np.zeros_like(obs))
            self.act_stack.append(0)
            self.rew_stack.append(0)
            self.done_stack.append(1)
            self.info_stack.append(None)

    def _get_observation(self):
        """Return current episode's N-stacked observation.

        For N=3, the first observation of the episode (reset) looks like:

        *= hasn't happened yet.

        GOAL  OBS  ACT  REW  DONE
        =========================
        g0    0    0.   0.   True
        g0    0    0.   0.   True
        g0    x0   0.   0.   False

        After the first step(a0) taken, yielding x1, r0, done0, info0, the next
        observation looks like:

        GOAL  OBS  ACT  REW  DONE
        =========================
        g0    0    0.   0.   True
        g0    x0   0.   0.   False
        g1    x1   a0   r0   d0

        A more chronologically intuitive way to re-order the column data would be:

        PREV_ACT  PREV_REW  PREV_DONE CURR_GOAL CURR_OBS
        ================================================
        0.        0.        True      g0        0
        0.        0.        False*    g0        x0
        a0        r0        info0     g1        x1

        Returns:
          episode_history: np.ndarray of observation.
        """
        episode_history = {
            'observations': np.stack(self.obs_stack, axis=0),
            'actions': np.stack(self.act_stack, axis=0),
            'rewards': np.stack(self.rew_stack, axis=0),
        }
        if self.is_goal_conditioned:
            episode_history['returns-to-go'] = np.stack(
                self.goal_stack, axis=0)
        return episode_history

    def reset(self):
        """Resets env and returns new observation."""
        obs = self._env.reset()
        # Create a N-1 "done" past frames.
        self.pad_current_episode(obs, self.num_stack_frames-1)
        # Create current frame (but with placeholder actions and rewards).
        if self.is_goal_conditioned:
            self.goal_stack.append(self._env.goal)
        self.obs_stack.append(obs)
        self.act_stack.append(0)
        self.rew_stack.append(0)
        self.done_stack.append(0)
        self.info_stack.append(None)
        return self._get_observation()

    def step(self, action: np.ndarray):
        """Replaces env observation with fixed length observation history."""
        # Update applied action to the previous timestep.
        self.act_stack[-1] = action
        obs, rew, done, info = self._env.step(action)
        self.rew_stack[-1] = rew
        # Update frame stack.
        self.obs_stack.append(obs)
        self.act_stack.append(0)  # Append unknown action to current timestep.
        self.rew_stack.append(0)
        self.info_stack.append(info)
        if self.is_goal_conditioned:
            self.goal_stack.append(self._env.goal)
        if done:
            if self.is_goal_conditioned:
                # rewrite the observations to reflect hindsight RtG conditioning.
                self.replace_goals_with_hindsight()
        return self._get_observation(), rew, done, info

    def replace_goals_with_hindsight(self):
        # We perform this after rew_stack has been updated.
        assert self.is_goal_conditioned
        window_return = sum(list(self.rew_stack))
        for r in self.rew_stack:
            self.goal_stack.append(window_return)
            window_return -= r


def build_env_fn(game_name):
    """Returns env constructor fn."""

    def env_fn():
        env = AtariEnvWrapper(game_name)
        env = SequenceEnvironmentWrapper(env, 4)
        return env

    return env_fn
