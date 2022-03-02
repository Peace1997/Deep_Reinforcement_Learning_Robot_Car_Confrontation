'''
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
import random
import argparse

import numpy as np
import torch



from simulator import ICRABattleField

from agent.core import MLPActorCritic
import time
parser = argparse.ArgumentParser(
    description="Test the trained model in the ICRA 2019 Battlefield")
parser.add_argument("--seed", type=int, default=233, help="Random seed")
parser.add_argument("--enemy", type=str, default="hand",
                    help="The opposite agent type [AC, hand]")
parser.add_argument("--load_model", action = 'store_true', help = "Whether to load the trained model",default=False)
parser.add_argument("--load_model_path", type = str, default = "ICRA_save.model", help = "The path of trained model")
parser.add_argument("--epoch", type=int, default=50,
                    help="Number of epoches to test")
args = parser.parse_args()


torch.random.manual_seed(args.seed)
torch.cuda.random.manual_seed(args.seed)
np.random.seed(args.seed)
#random.seed(args.seed)

env = ICRABattleField()
env.seed(args.seed)

obs_dim = env.observation_space.shape
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

red_agent = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=(256,256,256,256))
red_agent.red_load()

# blue_agent = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=(256,256,256,256))
# blue_agent.blue_load()

start_time = time.time()
def get_action(agent ,o, noise_scale):
    a = agent.act(torch.as_tensor(o, dtype=torch.float32))
    a += noise_scale * np.random.randn(act_dim)
    return np.clip(a, -act_limit, act_limit)

def test_agent():
    max_ep_len = 2000
    for j in range(3000):
        red_o, blue_o = env.reset()
        d, ep_ret, ep_len = False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time (noise_scale=0)
            # blue_a = env.action_space.sample() get_action(blue_agent,blue_o,0)

            red_o,blue_o, r, d, _ = env.step(get_action(red_agent,red_o,0),env.action_space.sample(),ep_len)
            env.render()
            ep_ret += r
            ep_len += 1
        print(ep_ret)

test_agent()
env.close()
end_time = time.time()
print('total time is :',end_time-start_time,'s')