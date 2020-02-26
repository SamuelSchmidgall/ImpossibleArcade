import math
import torch
import pickle
from scipy import signal
import numpy as np
from torch import nn
from torch.distributions import Normal



class ProximalPolicyOptimization(nn.Module):
    """
    Proximal Policy Optimization (https://arxiv.org/pdf/1707.06347.pdf) training module
    On-policy method of training actor-critic through clipping-based TRPO approximation
    """
    def __init__(self, actor_critic, epochs=10, minibatch_size=1000, timestep_size=4000, entropy_coefficient=0.01):
        super(ProximalPolicyOptimization, self).__init__()

        # generalized advantage estimation lambda
        self.lamb = 0.95
        # discount factor for discounted sum of rewards
        self.gamma = 0.99
        # ppo clipping ratio parameter
        self.ppo_clip = 0.2

        # number of minibatch SGD updates
        self.epochs = epochs
        # actor critic to optimize
        self.actor_critic = actor_critic
        # number of timesteps until gradient update
        self.timestep_size = timestep_size
        # minibatch size for gradient updates
        self.minibatch_size = minibatch_size
        # entropy coefficient -- higher value promotes more random actions
        self.entropy_coefficient = entropy_coefficient


    def forward(self, x, memory, evaluate=False, visual_obs=None):
        if visual_obs is None:
            action_mean, action_log_std = self.actor_critic(x)
        else:
            action_mean, action_log_std = self.actor_critic(x, visual_obs)

        action_std = torch.exp(action_log_std)

        distribution = Normal(loc=action_mean, scale=action_std)
        action = distribution.sample()
        log_probabilities = distribution.log_prob(action)
        log_probabilities = torch.sum(log_probabilities, dim=1)

        memory.log_probs.append(log_probabilities.detach())

        if evaluate:
            return action_mean.detach(), None
        return action, memory

    def entropy(self, action_log_std):
        dist_entropy = 0.5 + 0.5 * torch.log(2 * torch.ones(1)*math.pi) + action_log_std
        dist_entropy = dist_entropy.sum(-1).mean()
        return dist_entropy

    def evaluate(self, x, old_action, visual_obs=None):
        if visual_obs is not None:
            action_mean, action_log_std = self.actor_critic(x, visual_obs)
        else:
            action_mean, action_log_std = self.actor_critic(x)
        action_std = torch.exp(action_log_std)

        distribution = Normal(loc=action_mean, scale=action_std)
        log_probabilities = distribution.log_prob(old_action.squeeze(dim=1))
        log_probabilities = torch.sum(log_probabilities, dim=1)

        entropy = distribution.entropy()

        return log_probabilities, entropy

    def generalized_advantage_estimation(self, r, v, mask):
        batchsz = v.size(0)

        # v_target is worked out by Bellman equation.
        delta = torch.Tensor(batchsz)
        v_target = torch.Tensor(batchsz)
        adv_state = torch.Tensor(batchsz)

        prev_v = 0
        prev_v_target = 0
        prev_adv_state = 0
        for t in reversed(range(batchsz)):
            # mask here indicates a end of trajectory
            # this value will be treated as the target value of value network.
            # mask = 0 means the immediate reward is the real V(s) since it's end of trajectory.
            # formula: V(s_t) = r_t + gamma * V(s_t+1)
            v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]

            # please refer to : https://arxiv.org/abs/1506.02438
            # for generalized adavantage estimation
            # formula: delta(s_t) = r_t + gamma * V(s_t+1) - V(s_t)
            delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]

            # formula: A(s, a) = delta(s_t) + gamma * lamda * A(s_t+1, a_t+1)
            # here use symbol tau as lambda, but original paper uses symbol lambda.
            adv_state[t] = delta[t] + self.gamma * self.lamb * prev_adv_state * mask[t]

            # update previous
            prev_v_target = v_target[t]
            prev_v = v[t]
            prev_adv_state = adv_state[t]

        # normalize adv_state
        adv_state = (adv_state - adv_state.mean()) / (adv_state.std() + 1e-6)

        return adv_state, v_target


    def learn(self, memory):
        replay_len = len(memory.rewards)
        minibatch_count = self.timestep_size / self.minibatch_size

        if self.actor_critic.net_type == "vision":
            visual_states = torch.stack(memory.visual_states).unsqueeze(1)
            values = self.actor_critic.value(torch.FloatTensor(memory.sensor_states), visual_obs=visual_states).detach()
        else:
            values = self.actor_critic.value(torch.FloatTensor(memory.sensor_states)).detach()

        advantages, value_target = self.generalized_advantage_estimation(
            torch.FloatTensor(memory.rewards).unsqueeze(1), values, torch.FloatTensor(memory.reset_flags).unsqueeze(1))

        advantages = advantages.detach().numpy()
        value_target = value_target.detach().numpy()

        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for _ in range(self.epochs):
            minibatch_indices = list(range(replay_len))
            np.random.shuffle(minibatch_indices)
            minibatches = [minibatch_indices[int(_ * (replay_len/minibatch_count)):
                int((_ + 1) * (replay_len/minibatch_count))] for _ in range(int(minibatch_count))]

            for batch in minibatches:

                mb_states = torch.FloatTensor(np.array(memory.sensor_states)[batch])
                mb_actions = torch.stack(memory.actions).index_select(0, torch.LongTensor(batch))
                mb_old_log_probabilities = torch.stack(memory.log_probs).index_select(0, torch.LongTensor(batch))

                if self.actor_critic.net_type == "vision":
                    mb_image_states = torch.stack(memory.visual_states).index_select(0, torch.LongTensor(batch)).unsqueeze(1)

                if self.actor_critic.net_type == "vision":
                    predicted_values = self.actor_critic.value(mb_states, visual_obs=mb_image_states)
                else:
                    predicted_values = self.actor_critic.value(mb_states)

                if self.actor_critic.net_type == "vision":
                    log_probabilities, entropy = self.evaluate(mb_states, mb_actions, visual_obs=mb_image_states)
                else:
                    log_probabilities, entropy = self.evaluate(mb_states, mb_actions)

                mb_advantages = torch.FloatTensor(advantages[batch])

                ratio = (log_probabilities - mb_old_log_probabilities.squeeze()).exp()
                min_adv = torch.where(mb_advantages > 0,
                    (1 + self.ppo_clip) * mb_advantages, (1 - self.ppo_clip) * mb_advantages)
                policy_loss = -(torch.min(ratio * mb_advantages, min_adv)).mean() - self.entropy_coefficient*entropy.mean()

                value_loss = (torch.FloatTensor(value_target[batch]) - predicted_values.squeeze()).pow(2).mean()
                self.actor_critic.optimize(policy_loss, value_loss)

        #print(value_loss, policy_loss)



from copy import deepcopy

def run(agent, env):
    net_id = 1
    torch.set_num_threads(1)
    from Learning.Networks.ppo_networks import ReplayMemory


    timesteps = 0
    total_timesteps = 0
    max_timesteps = 10000000
    avg_action_magnitude = 0
    agent_replay = ReplayMemory()

    episode_itr = 0
    avg_sum_rewards = 0.0

    while total_timesteps < max_timesteps:

        episode_itr += 1
        game_over = False

        sensor_obs = env.reset()
        while not game_over:

            local_action, agent_replay = agent(
                x=torch.FloatTensor(sensor_obs).unsqueeze(0), memory=agent_replay)

            agent_replay.sensor_states.append(deepcopy(sensor_obs))
            agent_replay.actions.append(local_action)

            local_action = local_action.squeeze(dim=1).numpy()
            sensor_obs, reward, game_over, information = env.step(np.clip(local_action, a_min=-1, a_max=1))
            agent_replay.reset_flags.append(0 if game_over else 1)

            agent_replay.rewards.append(reward)

            avg_sum_rewards += reward

            timesteps += 1
            total_timesteps += 1

        if timesteps > agent.timestep_size:
            agent.learn(memory=agent_replay)

            avg_action_magnitude /= timesteps

            print("Time: {} Reward: {}, Timestep: {}".format(
                round(timesteps/episode_itr, 8), round(avg_sum_rewards/episode_itr, 8), total_timesteps))

            timesteps = 0
            episode_itr = 0
            avg_sum_rewards = 0.0
            avg_action_magnitude  = 0

            with open("saved_model_{}_{}.pkl".format("linear", net_id), "wb") as f:
                pickle.dump(agent, f)

            with open("saved_model_{}_{}.pkl".format("linear", net_id), "wb") as f:
                pickle.dump([agent_replay.rewards, agent_replay.reset_flags], f)

            agent_replay.clear()



