"""Example of using policy evaluator classes directly to implement training.
Instead of using the built-in Trainer classes provided by RLlib, here we define
a custom PolicyGraph class and manually coordinate distributed sample
collection and policy optimization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gym

import ray
from ray import tune
from ray.rllib.evaluation import PolicyGraph, PolicyEvaluator, SampleBatch
from ray.rllib.evaluation.metrics import collect_metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import scipy.signal

from torch.autograd import Variable
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--num-iters", type=int, default=20)
parser.add_argument("--num-workers", type=int, default=2)

class TRPOAgent(nn.Module):
    def __init__(self, state_shape, n_actions, hidden_size=32):
        '''
        Here you should define your model
        You should have LOG-PROBABILITIES as output because you will need it to compute loss
        We recommend that you start simple:
        use 1-2 hidden layers with 100-500 units and relu for the first try
        '''
        nn.Module.__init__(self)

        self.n_actions = n_actions
        self.state_hape = state_shape

        self.model = nn.Sequential(
            nn.Linear(state_shape[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.LogSoftmax()
        )

    def forward(self, states):
        """
        takes agent's observation (Variable), returns log-probabilities (Variable)
        :param state_t: a batch of states, shape = [batch_size, state_shape]
        """

        # Use your network to compute log_probs for given state
        log_probs = self.model(states)
        return log_probs

    def get_log_probs(self, states):
        '''
        Log-probs for training
        '''

        return self.forward(states)

    def get_probs(self, states):
        '''
        Probs for interaction
        '''

        return torch.exp(self.forward(states))

    def act(self, obs, sample=True):
        '''
        Samples action from policy distribution (sample = True) or takes most likely action (sample = False)
        :param: obs - single observation vector
        :param sample: if True, samples from \pi, otherwise takes most likely action
        :returns: action (single integer) and probabilities for all actions
        '''

        probs = self.get_probs(Variable(torch.FloatTensor([obs]))).data.numpy()

        if sample:
            action = int(np.random.choice(self.n_actions, p=probs[0]))
        else:
            action = int(np.argmax(probs))

        return action, probs[0]



class CustomPolicy(PolicyGraph):
    """Example of a custom policy graph written from scratch.
    You might find it more convenient to extend TF/TorchPolicyGraph instead
    for a real policy.
    """

    def __init__(self, observation_space, action_space, config):
        PolicyGraph.__init__(self, observation_space, action_space, config)
        # example parameter
        self.w = 1.0
        self.observation_shape = observation_space.shape
        self.n_actions = action_space.n
        self.agent = TRPOAgent(self.observation_shape, self.n_actions)


    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return random actions
        return [self.agent.act(obs)[0] for obs in obs_batch], [], {}

    def get_flat_params_from(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))

        flat_params = torch.cat(params)
        return flat_params

    def set_flat_params_to(self, model, flat_params):
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size



    def get_cummulative_returns(r, gamma=1):
        """
        Computes cummulative discounted rewards given immediate rewards
        G_i = r_i + gamma*r_{i+1} + gamma^2*r_{i+2} + ...
        Also known as R(s,a).
        """
        r = np.array(r)
        assert r.ndim >= 1
        return scipy.signal.lfilter([1], [1, -gamma], r[::-1], axis=0)[::-1]

    def get_loss(agent, observations, actions, cummulative_returns, old_probs):
        """
        Computes TRPO objective
        :param: observations - batch of observations
        :param: actions - batch of actions
        :param: cummulative_returns - batch of cummulative returns
        :param: old_probs - batch of probabilities computed by old network
        :returns: scalar value of the objective function
        """
        batch_size = observations.shape[0]
        log_probs_all = agent.get_log_probs(observations)
        probs_all = torch.exp(log_probs_all)

        probs_for_actions = probs_all[torch.arange(
            0, batch_size, out=torch.LongTensor()), actions]
        old_probs_for_actions = old_probs[torch.arange(
            0, batch_size, out=torch.LongTensor()), actions]

        # Compute surrogate loss, aka importance-sampled policy gradient
        Loss = -torch.mean(cummulative_returns * (probs_for_actions / old_probs_for_actions))

        return Loss

    def get_kl(agent, observations, actions, cummulative_returns, old_probs_all):
        """
        Computes KL-divergence between network policy and old policy
        :param: observations - batch of observations
        :param: actions - batch of actions
        :param: cummulative_returns - batch of cummulative returns (we don't need it actually)
        :param: old_probs - batch of probabilities computed by old network
        :returns: scalar value of the KL-divergence
        """
        batch_size = observations.shape[0]
        log_probs_all = agent.get_log_probs(observations)
        probs_all = torch.exp(log_probs_all)

        # Compute Kullback-Leibler divergence (see formula above)
        # Note: you need to sum KL and entropy over all actions, not just the ones agent took
        old_log_probs_all = torch.log(old_probs_all + 1e-10)

        kl = torch.sum(old_probs_all * (old_log_probs_all - log_probs_all)) / batch_size

        return kl

    def get_entropy(agent, observations):
        """
        Computes entropy of the network policy
        :param: observations - batch of observations
        :returns: scalar value of the entropy
        """

        observations = Variable(torch.FloatTensor(observations))

        batch_size = observations.shape[0]
        log_probs_all = agent.get_log_probs(observations)
        probs_all = torch.exp(log_probs_all)

        entropy = torch.sum(-probs_all * log_probs_all) / batch_size

        return entropy

    def linesearch(f, x, fullstep, max_kl):
        """
        Linesearch finds the best parameters of neural networks in the direction of fullstep contrainted by KL divergence.
        :param: f - function that returns loss, kl and arbitrary third component.
        :param: x - old parameters of neural network.
        :param: fullstep - direction in which we make search.
        :param: max_kl - constraint of KL divergence.
        :returns:
        """
        max_backtracks = 10
        loss, _, = f(x)
        for stepfrac in .5 ** np.arange(max_backtracks):
            xnew = x + stepfrac * fullstep
            new_loss, kl = f(xnew)
            actual_improve = new_loss - loss
            if kl.data.numpy() <= max_kl and actual_improve.data.numpy() < 0:
                x = xnew
                loss = new_loss
        return x

    def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
        """
        This method solves system of equation Ax=b using iterative method called conjugate gradients
        :f_Ax: function that returns Ax
        :b: targets for Ax
        :cg_iters: how many iterations this method should do
        :residual_tol: epsilon for stability
        """
        p = b.clone()
        r = b.clone()
        x = torch.zeros(b.size())
        rdotr = torch.sum(r * r)
        for i in range(cg_iters):
            z = f_Ax(p)
            v = rdotr / (torch.sum(p * z) + 1e-8)
            x += v * p
            r -= v * z
            newrdotr = torch.sum(r * r)
            mu = newrdotr / (rdotr + 1e-8)
            p = r + mu * p
            rdotr = newrdotr
            if rdotr < residual_tol:
                break
        return x


    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}


    def get_weights(self):
        return self.get_flat_params_from(self.agent)

    def set_weights(self, weights):
        self.set_flat_params_to(self.agent)


def training_workflow(config, reporter):
    # Setup policy and policy evaluation actors
    env = gym.make("CartPole-v0")
    policy = CustomPolicy(env.observation_space, env.action_space, {})
    workers = [
        PolicyEvaluator.as_remote().remote(lambda c: gym.make("CartPole-v0"),
                                           CustomPolicy)
        for _ in range(config["num_workers"])
    ]

    for _ in range(config["num_iters"]):
        # Broadcast weights to the policy evaluation workers
        weights = ray.put({"default_policy": policy.get_weights()})
        for w in workers:
            w.set_weights.remote(weights)

        # Gather a batch of samples
        T1 = SampleBatch.concat_samples(
            ray.get([w.sample.remote() for w in workers]))
        print("DEBUG* BATCH ************************")
        print(T1)
        print("DEBUG*************************")




        # Improve the policy using the T1 batch
        policy.learn_on_batch(T1)

        reporter(**collect_metrics(remote_evaluators=workers))


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    tune.run(
        training_workflow,
        resources_per_trial={
            "gpu": 1 if args.gpu else 0,
            "cpu": 1,
            "extra_cpu": args.num_workers,
        },
        config={
            "num_workers": args.num_workers,
            "num_iters": args.num_iters,
        },
)