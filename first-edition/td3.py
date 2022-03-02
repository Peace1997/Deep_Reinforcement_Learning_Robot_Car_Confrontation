from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import agent.core as core
from simulator import ICRABattleField
from torch.utils.tensorboard import SummaryWriter
import os
import itertools


class ReplayBuffer:
    """
    red_a simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



def td3(actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=200300, epochs=100, replay_size=int(2e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-4, q_lr=1e-3, batch_size=100, start_steps=2000000, 
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=3000, 
        save_freq=1):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)
    Args:
        env_fn : red_a function which creates red_a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for red_a PyTorch Module with an ``act`` 
            method, red_a ``pi`` module, red_a ``q1`` module, and red_a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept red_a batch 
            of observations and red_a batch of actions as inputs. When called, 
            these should return:
            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to TD3.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs to run and train agent.
        replay_size (int): Maximum length of replay buffer.
        gamma (float): Discount factor. (Always between 0 and 1.)
        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:
            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)
        pi_lr (float): Learning rate for policy.
        q_lr (float): Learning rate for Q-networks.
        batch_size (int): Minibatch size for SGD.
        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.
        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.
        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.
        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)
        target_noise (float): Stddev for smoothing noise added to target 
            policy.
        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.
        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.
        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    writer = SummaryWriter('./path/to/log')
    env, test_env = ICRABattleField(), ICRABattleField()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    red_ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    # red_ac.red_load()
    # blue_ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    # blue_ac.blue_load()

    print(red_ac)
    ac_targ = deepcopy(red_ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(red_ac.q1.parameters(), red_ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get red_a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [red_ac.pi, red_ac.q1, red_ac.q2])
    
    # Set up function for computing TD3 Q-losses  与DDPG主要不同点
    def compute_loss_q(data):
        red_o, red_a, r, red_o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = red_ac.q1(red_o,red_a)
        q2 = red_ac.q2(red_o,red_a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.pi(red_o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            red_a2 = pi_targ + epsilon
            red_a2 = torch.clamp(red_a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(red_o2, red_a2)
            q2_pi_targ = ac_targ.q2(red_o2, red_a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        red_o = data['obs']
        q1_pi = red_ac.q1(red_o, red_ac.pi(red_o))
        return -q1_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(red_ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True


            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(red_ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
        return loss_q

    def get_action(ac,o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            red_o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                red_o, r, d, _ = test_env.step(get_action(red_o, 0))
                ep_ret += r
                ep_len += 1

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    red_o, blue_o  = env.reset()
    ep_ret, ep_len = 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from red_a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            red_a = get_action(red_ac,red_o, act_noise)
        else:
            red_a = env.action_space.sample()

        #blue_a = get_action(blue_ac,blue_o,act_noise)
        blue_a = env.action_space.sample()

        # Step the env
        red_o2, blue_o ,r, d, _ = env.step(red_a,blue_a,ep_len)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(red_o, red_a, r, red_o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        red_o = red_o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            writer.add_scalar('Return', ep_ret, t)
            print("times:",t,"/",total_steps,"Return:",ep_ret)
            red_o, blue_o = env.reset()
            ep_ret, ep_len = 0, 0
        if t % 2000000 ==0:
            os.makedirs('./model1_8_2/'+str(int(t/2000000)))
            red_ac.save('./model1_8_2/'+str(int(t/2000000))+'/')
        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                loss_q = update(data=batch,timer=j)
            writer.add_scalar('Loss Q', loss_q, t)
            #writer.add_scalar('Loss Pi', loss_pi, t)
        #env.render() 

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='td3')
    args = parser.parse_args()


    td3(actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        )