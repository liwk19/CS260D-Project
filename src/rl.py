#!/usr/bin/env python3
from config import FLAGS
import gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

from lib import common

# below is a temp solution: https://github.com/openai/spinningup/issues/16
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# "conda install nomkl" solves the above issue


GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 4
NUM_ENVS = 2

REWARD_STEPS = 2
CLIP_GRAD = 0.1





MAX_NODE = 100
MAX_EDGE = 101

def rl_main(dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default='xxx', help="Name of the run")
    args = parser.parse_args()
    device = FLAGS.device

    # make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))
    # envs = [make_env() for _ in range(NUM_ENVS)]

    envs = _make_env()
    writer = SummaryWriter(comment="-pong-a2c_" + args.name)

    net = MyNet(55, MAX_NODE).to(device) # use double() to avoid float-double error
    print(net)

    agent = MyPolicyAgent(lambda x: net(x)[0], apply_softmax=True,
                                   device=device, preprocessor=my_states_preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []

    with common.RewardTracker(writer, stop_reward=18) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
                batch.clear()

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_actions_v_sum = 0.0
                entropy_loss_v_sum = 0.0
                adv_v = vals_ref_v - value_v.squeeze(-1).detach()
                assert type(logits_v) is list
                for i, p in enumerate(logits_v):
                    log_prob_v = F.log_softmax(p, dim=1)
                    prob_v = F.softmax(p, dim=1)
                    # x = adv_v[i]
                    # y = log_prob_v[0,actions_t[i]]
                    log_prob_actions_v_sum += adv_v[i] * log_prob_v[0,actions_t[i]]
                    entropy_loss_v_sum += (prob_v * log_prob_v).sum(dim=1)
                loss_policy_v = -log_prob_actions_v_sum / len(logits_v)
                # log_prob_v = F.log_softmax(logits_v, dim=1)
                # log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                # loss_policy_v = -log_prob_actions_v.mean()
                entropy_loss_v = ENTROPY_BETA * entropy_loss_v_sum / len(logits_v)
                # prob_v = F.softmax(logits_v, dim=1)
                # entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                # apply entropy and value gradients
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                # get full loss
                loss_v += loss_policy_v

                print(f'Iter loss_entropy {entropy_loss_v} loss_policy {loss_policy_v}'
                      f' loss_value {loss_value_v} loss_total {loss_v}')
                tb_tracker.track("advantage",       adv_v, step_idx)
                tb_tracker.track("values",          value_v, step_idx)
                tb_tracker.track("batch_rewards",   vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy",    entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy",     loss_policy_v, step_idx)
                tb_tracker.track("loss_value",      loss_value_v, step_idx)
                tb_tracker.track("loss_total",      loss_v, step_idx)
                tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var",        np.var(grads), step_idx)


def _make_env():
    from data import load_all_gs, load_encoders
    all_gs = load_all_gs(remove_all_pragma_nodes=True)
    encoders = load_encoders()
    # enc_ntype, enc_ptype = encoders['enc_ntype'], encoders['enc_ptype']
    print(f'Loaded {len(all_gs)} gs')
    rtn = []
    for i in range(FLAGS.num_envs):
        g_to_select = all_gs[i % len(all_gs)]
        # print(g)
        env = MyEnv(g_to_select, encoders)
        rtn.append(env)
    return rtn




import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np



import networkx as nx
import copy

class MyEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, g, encoders):
        super(MyEnv, self).__init__()

        self.graph = g
        self.orig_graph = copy.deepcopy(g)
        self.encoders = encoders

        # self.max_node = MAX_NODE
        # self.action_space = gym.spaces.MultiDiscrete([self.max_node])
        # self.observation_space = {}
        # self.observation_space['adj'] = gym.Space(shape=[self.max_node, self.max_node])
        # self.observation_space['node'] = gym.Space(shape=[self.max_node, 1])

        # self.df = df
        # self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        #
        # # Actions of the format Buy x%, Sell x%, Hold, etc.
        # self.action_space = spaces.Box(=
        #     low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        #
        # # Prices contains the OHCL values for the last five prices
        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=(6, 6), dtype=np.float16)

    def _get_observation(self):


        from data import encode_g_torch

        X, edge_index = encode_g_torch(
            self.graph, self.encoders['enc_ntype'], self.encoders['enc_ptype'])

        ob = {}
        # if self.is_normalize:
        #     E = self.normalize_adj(E)
        ob['X'] = X
        ob['edge_index'] = edge_index
        return ob

        # """
        # :return: ob, where ob['adj'] is E with dim b x n x n and ob['node']
        # is F with dim 1 x n x m. NB: n = node_num + node_type_num
        # """
        # n = self.graph.number_of_nodes()
        # F = np.zeros((self.max_node, 1))
        # F[n+1,0] = 1
        #
        # E = np.zeros((self.max_node, self.max_node))
        # E[n,:n] = np.asarray(nx.to_numpy_matrix(self.graph))
        # # E[n+1,:n+1] += np.eye(n+1)
        #
        # ob = {}
        # # if self.is_normalize:
        # #     E = self.normalize_adj(E)
        # ob['adj'] = E
        # ob['node'] = F
        # return ob

    #     # Get the stock data points for the last 5 days and scale to between 0-1
    #     frame = np.array([
    #         self.df.loc[self.current_step: self.current_step +
    #                     5, 'Open'].values / MAX_SHARE_PRICE,
    #         self.df.loc[self.current_step: self.current_step +
    #                     5, 'High'].values / MAX_SHARE_PRICE,
    #         self.df.loc[self.current_step: self.current_step +
    #                     5, 'Low'].values / MAX_SHARE_PRICE,
    #         self.df.loc[self.current_step: self.current_step +
    #                     5, 'Close'].values / MAX_SHARE_PRICE,
    #         self.df.loc[self.current_step: self.current_step +
    #                     5, 'Volume'].values / MAX_NUM_SHARES,
    #     ])
    #
    #     # Append additional data and scale each value to between 0-1
    #     obs = np.append(frame, [[
    #         self.balance / MAX_ACCOUNT_BALANCE,
    #         self.max_net_worth / MAX_ACCOUNT_BALANCE,
    #         self.shares_held / MAX_NUM_SHARES,
    #         self.cost_basis / MAX_SHARE_PRICE,
    #         self.total_shares_sold / MAX_NUM_SHARES,
    #         self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
    #     ]], axis=0)
    #
    #     return obs
    #
    # def _take_action(self, action):
    #     # Set the current price to a random price within the time step
    #     current_price = random.uniform(
    #         self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])
    #
    #     action_type = action[0]
    #     amount = action[1]
    #
    #     if action_type < 1:
    #         # Buy amount % of balance in shares
    #         total_possible = int(self.balance / current_price)
    #         shares_bought = int(total_possible * amount)
    #         prev_cost = self.cost_basis * self.shares_held
    #         additional_cost = shares_bought * current_price
    #
    #         self.balance -= additional_cost
    #         self.cost_basis = (
    #             prev_cost + additional_cost) / (self.shares_held + shares_bought)
    #         self.shares_held += shares_bought
    #
    #     elif action_type < 2:
    #         # Sell amount % of shares held
    #         shares_sold = int(self.shares_held * amount)
    #         self.balance += shares_sold * current_price
    #         self.shares_held -= shares_sold
    #         self.total_shares_sold += shares_sold
    #         self.total_sales_value += shares_sold * current_price
    #
    #     self.net_worth = self.balance + self.shares_held * current_price
    #
    #     if self.net_worth > self.max_net_worth:
    #         self.max_net_worth = self.net_worth
    #
    #     if self.shares_held == 0:
    #         self.cost_basis = 0

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        ### init
        info = {}  # info we care about
        self.graph_old = copy.deepcopy(self.graph)
        total_nodes = self.graph.number_of_nodes()

        ### take action
        # if action[0, 3] == 0:   # not stop
        #     stop = False
        #     if action[0, 1] >= total_nodes:
        #         self.graph.add_node(int(action[0, 1]))
        #         self._add_edge(action)
        #     else:
        #         self._add_edge(action)  # add new edge
        # else:   # stop
        #     stop = True

        ### calculate intermediate rewards
        # # todo: add neccessary rules for the task
        # if self.graph.number_of_nodes() + self.graph.number_of_edges()-self.graph_old.number_of_nodes() - \
        #     self.graph_old.number_of_edges() > 0:
        #     reward_step = self.reward_step_total / self.max_node
        #     # successfully added node/edge
        # else:
        #     reward_step = -self.reward_step_total / self.max_node # edge
        #     self.graph = self.graph_old
        #     # already exists
        #
        # ### calculate and use terminal reward
        # if self.graph.number_of_nodes() >= self.max_node - 1 or self.counter >= self.max_action or stop:
        #
        #     # property rewards
        #     ## todo: add property reward
        #     reward_terminal = 1 # arbitrary choice
        #
        #     new = True  # end of episode
        #     reward = reward_step + reward_terminal
        #
        #     # print terminal graph information
        #     info['final_stat'] = reward_terminal
        #     info['reward'] = reward
        #     info['stop'] = stop
        # ### use stepwise reward
        # else:
        #     new = False
        #     reward = reward_step
        #
        # # get observation
        # ob = self.get_observation()
        #
        # self.counter += 1
        # if new:
        #     self.counter = 0

        ob = self._get_observation()
        reward = 77
        done = False

        return ob, reward, done, {}



        # Execute one time step within the environment
        # self._take_action(action)
        #
        # self.current_step += 1
        #
        # if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
        #     self.current_step = 0
        #
        # delay_modifier = (self.current_step / MAX_STEPS)
        #
        # reward = self.balance * delay_modifier
        # done = self.net_worth <= 0
        #
        # obs = self._next_observation()
        #
        # return obs, reward, done, {}

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        print('Reset@@@@@')
        # self.graph.clear()
        self.graph = self.orig_graph

        # self.graph.add_node(0)
        # self.counter = 0
        ob = self._get_observation()
        return ob

        # Reset the state of the environment to an initial state
        # self.balance = INITIAL_ACCOUNT_BALANCE
        # self.net_worth = INITIAL_ACCOUNT_BALANCE
        # self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        # self.shares_held = 0
        # self.cost_basis = 0
        # self.total_shares_sold = 0
        # self.total_sales_value = 0
        #
        # # Set the current step to a random point within the data frame
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'Open'].values) - 6)

        # return self._next_observation()

    def render(self, mode='human', close=False):
        print('Rendering!!!!!@@@@@')
        # Render the environment to the screen
        # profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        #
        # print(f'Step: {self.current_step}')
        # print(f'Balance: {self.balance}')
        # print(
        #     f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        # print(
        #     f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        # print(
        #     f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        # print(f'Profit: {profit}')


def my_states_preprocessor(states):
    from torch_geometric.data import Data, Batch

    data_list = []
    for state in states:
        data_list.append(Data(
            x=state['X'],
            edge_index=state['edge_index']
        ))

    return Batch.from_data_list(data_list).to(FLAGS.device)


    # """
    # Convert list of states into the form suitable for model. By default we assume Variable
    # :param states: list of numpy arrays with states
    # :return: Variable
    # """
    # assert type(states) is list
    # rtn = {}
    # keys = states[0].keys()
    # for k in keys:
    #     rtn[k] = np.array([np.array(s[k], copy=False) for s in states], copy=False)
    #     rtn[k] = torch.FloatTensor(rtn[k]).to(FLAGS.device)
    # return rtn
    # if len(states) == 1:
    #     np_states = np.expand_dims(states[0], 0)
    # else:
    #     np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    # return torch.tensor(np_states)


from utils import MLP, OurTimer
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool


from ptan.agent import PolicyAgent

class MyPolicyAgent(PolicyAgent):
    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        """
        Return actions from given list of states
        :param states: list of states
        :return: list of actions
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v = self.model(states)
        actions = []
        if self.apply_softmax:
            assert type(probs_v) is list
            for p in probs_v:
                p = F.softmax(p, dim=1)
                probs = p.data.cpu().numpy()
                action = self.action_selector(probs)
                actions.append(action)
        return np.array(actions), agent_states

class MyNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(MyNet, self).__init__()
        # pass
        D = FLAGS.D

        self.conv1 = GATConv(59, D) # TODO: less hard coding
        self.conv2 = GATConv(D, D // 2)
        self.conv3 = GATConv(D // 2, D // 2)

        # if FLAGS.task == 'regression':
        self.out_dim = 1
            # self.loss_fucntion = torch.nn.MSELoss()
        # else:
        #     self.out_dim = 2
        #     self.loss_fucntion = torch.nn.CrossEntropyLoss()

        self.policy = MLP(D // 2, self.out_dim,
                         hidden_channels=[D // 2, D // 4, D // 8],
                         num_hidden_lyr=3)

        self.value = MLP(D // 2, self.out_dim,
                       hidden_channels=[D // 2, D // 4, D // 8],
                       num_hidden_lyr=3)


        # conv_out_size = self._get_conv_out(input_shape)
        # self.policy = nn.Sequential(
        #     nn.Linear(1, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1)
        # )

        # self.value = nn.Sequential(
        #     nn.Linear(1, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1)
        # )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = F.relu(self.conv1(x, edge_index))

        # out, edge_index, _, batch, perm, score = self.pool1(
        #     out, edge_index, None, batch)
        # ratio = out.size(0) / x.size(0)



        out = F.relu(self.conv2(out, edge_index))
        out = F.relu(self.conv3(out, edge_index))

        policy =self.policy(out)

        out = global_add_pool(out, batch)
        value = self.value(out)
        print()

        policy_list = []
        ind_list = _gen_ind_list_from_batch(batch)
        for (start, end) in ind_list:
            policy_list.append(policy[start:end].t())

        # exit()
        return policy_list, value


        # fx = x.float() / 256
        # conv_out = self.conv(fx).view(fx.size()[0], -1)
        # x = x['node'].view()
        # pi = self.policy(x['node'])
        # pi = pi.view((pi.shape[0], self.n_actions))
        #
        # gemb = torch.sum(x['node'], dim=1)
        # v = self.value(gemb)
        # v = v.view((v.shape[0], 1))
        # return pi, v


def _gen_ind_list_from_batch(batch):
    ind_list = []
    start = 0
    cur_x = 0
    length = batch.shape[0]
    for i, x in enumerate(batch):
        # print(i, x, cur_x, x.item() != cur_x)
        if x.item() != cur_x:
            assert x == cur_x + 1
            ind_list.append((start, i))
            start = i
            cur_x = x.item()
    ind_list.append((start, length))
    return ind_list


def unpack_batch(batch, net, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable

    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        # states.append(np.array(exp.state, copy=False))
        states.append(exp.state)
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            # last_states.append(np.array(exp.last_state, copy=False))
            last_states.append(exp.last_state)
    # states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    states_v = my_states_preprocessor(states)
    actions_t = torch.LongTensor(actions).to(device)
    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float64)
    if not_done_idx:
        # last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_states_v = my_states_preprocessor(last_states)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v

