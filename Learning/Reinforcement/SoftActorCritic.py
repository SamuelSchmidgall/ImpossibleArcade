import torch
from torch.optim import Adam
import torch.nn.functional as F
from Learning.Networks.utils import soft_update, hard_update, ReplayMemory
from Learning.Networks.networks_pytorch import QNetwork, SpinalNetworkH1Motor


class SAC(object):
    def __init__(self, num_inputs, action_space, args, act_space):

        self.tau = args["tau"]
        self.gamma = args["gamma"]
        self.alpha = args["alpha"]

        self.target_update_interval = args["target_update_interval"]
        self.automatic_entropy_tuning = args["automatic_entropy_tuning"]

        self.critic = QNetwork(num_inputs, action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=args["lr"])

        self.critic_target = QNetwork(num_inputs, action_space)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor((1, action_space))).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = Adam([self.log_alpha], lr=args["lr"])

        self.policy = SpinalNetworkH1Motor(num_inputs, action_space, act_space)
        self.policy_optim = Adam(self.policy.parameters(), lr=args["lr"])


    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


sac_hyperparams = {
    "seed":1,
    "lr": 0.0003,
    "tau": 0.005,
    "alpha": 0.2,
    "gamma": 0.99,
    "batch_size":256,
    "num_steps":1000001,
    "start_steps": 10000,
    "updates_per_step": 1,
    "replay_size": 1000000,
    "target_update_interval": 1,
    "automatic_entropy_tuning": False,
}



def train_sac():
    import gym
    env = gym.make("HalfCheetahHier-v2")
    agent = SAC(env.observation_space.shape[0],
        env.action_space.shape[0], sac_hyperparams, env.action_space)

    rewards_l = list()

    # replay buffer
    memory = ReplayMemory(sac_hyperparams["replay_size"])

    updates = 0
    total_numsteps = 0

    for i_episode in range(10000):
        done = False
        episode_steps = 0
        episode_reward = 0
        state = env.reset()

        while not done:
            if sac_hyperparams["start_steps"] > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > sac_hyperparams["batch_size"]:
                # Number of updates per step in environment
                for i in range(sac_hyperparams["updates_per_step"]):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = \
                        agent.update_parameters(memory, sac_hyperparams["batch_size"], updates)

                    updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

        if total_numsteps > sac_hyperparams["num_steps"]:
            break

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}"
              .format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        rewards_l.append(episode_reward)

        import pickle
        with open("model_dump.pkl", "wb") as f:
            pickle.dump(agent, f)

        with open("reward_dump.pkl", "wb") as f:
            pickle.dump(rewards_l, f)


#train_sac()
















































