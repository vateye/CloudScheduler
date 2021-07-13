import os
import time
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Manager
import sys

import warnings
warnings.filterwarnings("ignore")

sys.path.append('..')

from framework.instance import InstanceConfig
from framework.machine import MachineConfig

from framework.episode import Episode
from framework.trigger import ThresholdTrigger

from framework.DRL.agent import Agent
from framework.DRL.DRL import RLAlgorithm
from framework.DRL.policynet import PolicyNet
from framework.DRL.reward_giver import AverageCompletionRewardGiver, MakespanRewardGiver
from framework.DRL.utils import features_extract_func, features_normalize_func, multiprocessing_run

os.environ['CUDA_VISIBLE_DEVICES'] = ''

np.random.seed(41)
tf.random.set_random_seed(41)
# ************************ Parameters Setting Start ************************
machines_number = 5
jobs_len = 10
n_iter = 100
n_episode = 12
jobs_csv = '../jobs_files/jobs.csv'

policynet = PolicyNet(5)
reward_giver = MakespanRewardGiver(-1)
features_extract_func = features_extract_func
features_normalize_func = features_normalize_func

name = '%s-%s-m%d' % (reward_giver.name, policynet.name, machines_number)
model_dir = './agents/%s' % name
# ************************ Parameters Setting End ************************

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

agent = Agent(name, policynet, 1, reward_to_go=True, nn_baseline=True, normalize_advantages=True,
              model_save_path='%s/model.ckpt' % model_dir)

# machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
# csv_reader = CSVReader(jobs_csv)
# jobs_configs = csv_reader.generate(0, jobs_len)

machine_configs = [
    MachineConfig(0, 12, 20, 20),
    MachineConfig(1, 21, 25, 25)
]  # MachineConfigLoader('./data/machine_resources.a.csv')

instance_configs = [
    InstanceConfig(0, 0, 3, 5, 5, [3, 6, 5], [5, 5, 5]),
    InstanceConfig(1, 0, 3, 5, 5, [3, 2, 1], [5, 5, 5]),
    InstanceConfig(2, 1, 5, 5, 5, [5, 5, 5], [5, 5, 5]),
    InstanceConfig(3, 1, 5, 5, 5, [5, 5, 5], [5, 5, 5]),
    InstanceConfig(4, 1, 5, 5, 5, [5, 5, 5], [5, 5, 5])
]  # InstanceConfigLoader('./data/output_instance_deployed_a.csv')


for itr in range(n_iter):
    tic = time.time()
    print("********** Iteration %i ************" % itr)
    processes = []

    manager = Manager()
    trajectories = manager.list([])
    makespans = manager.list([])
    average_completions = manager.list([])
    average_slowdowns = manager.list([])
    for i in range(n_episode):
        algorithm = RLAlgorithm(agent, reward_giver, features_extract_func=features_extract_func,
                                features_normalize_func=features_normalize_func)
        trigger = ThresholdTrigger()
        episode = Episode(machine_configs, instance_configs, trigger, algorithm, None)
        algorithm.reward_giver.attach(episode.simulation)
        p = Process(target=multiprocessing_run,
                    args=(episode, trajectories, makespans))

        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    agent.log('makespan', np.mean(makespans), agent.global_step)

    toc = time.time()

    print(np.mean(makespans), toc - tic)

    all_observations = []
    all_actions = []
    all_rewards = []
    for trajectory in trajectories:
        observations = []
        actions = []
        rewards = []
        for node in trajectory:
            observations.append(node.observation)
            actions.append(node.action)
            rewards.append(node.reward)

        all_observations.append(observations)
        all_actions.append(actions)
        all_rewards.append(rewards)

    all_q_s, all_advantages = agent.estimate_return(all_rewards)

    agent.update_parameters(all_observations, all_actions, all_advantages)

agent.save()