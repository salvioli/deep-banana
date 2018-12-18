import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
import torch


def train_agent(agent, environment, n_episodes=800, max_t=1000,
                eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                stop_avg_score=13, score_window_size=100):
    """

    :param agent: agent to be trained
    :param environment: unity environment in which the agent should be trained
    :param n_episodes: maximum number of training episodes
    :param max_t: maximum timesteps per episode
    :param eps_start: starting value for epsilon
    :param eps_end: minimum value for epsilon
    :param eps_decay: multiplicative decay factor for epsilon used for exponential decay
    :param stop_avg_score: score at which the task is considered solved
    :param score_window_size: size of the moving average window for smoothing agent scores
    :return: trained agent
    """

    scores = []
    scores_window = deque(maxlen=score_window_size)

    brain_name = environment.brain_names[0]

    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        env_info = environment.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)

            env_info = environment.step(action)
            env_info = env_info[brain_name]

            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            score += reward
            state = next_state
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay* eps)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= stop_avg_score:
            solution_episodes_n = max(i_episode - 100, 0)
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
                  .format(solution_episodes_n, np.mean(scores_window)))
            break

    return agent, solution_episodes_n, np.mean(scores_window), scores


def demo_agent(agent, env):
    """
    Demonstrates the agent in the environment for one episode.
    :param agent:
    :param env:
    :return:
    """
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score

    while True:
        action = agent.act(state, eps=0)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break

    print("Score: {}".format(score))

    return score


def save_checkpoint(agent, file_name_tag):
    """
    Saves agents internal qnetwork weights to file in a file whose name includes the following information
    :param agent: agent whose weights should be saved
    :param file_name_tag: unique tag for the checkpoint file name
    :return checkpoint_filename: filename of the checkpoint file just saved
    """

    checkpoint_filename = 'checkpoints/checkpoint_' + file_name_tag + '.pth'
    torch.save(agent.qnetwork_local.state_dict(), checkpoint_filename)
    return checkpoint_filename


def generate_filename_tag(agent, avg_score, episodes_n):
    '''
    Returns the filename with standard formatting. Encodes in a string the following information:
    :param agent: agent to be encoded, used to extract type
    :param avg_score: average score at the end of training
    :param episodes_n: number of episodes that the agent took to solve the task
    :return: string with the encoded information passed as parameters
    '''
    file_name_tag = '{type}_[{fc1s},{fc2s}]_{score:.2f}_{episodes}_{tstamp}' \
        .format(type=agent.__class__.__name__,
                episodes=episodes_n,
                fc1s=agent.fc1_size,
                fc2s=agent.fc2_size,
                score=avg_score,
                tstamp=time.strftime("%Y%m%d-%H%M%S"))
    return file_name_tag


def plot_score(scores):
    """
    Plot and show the average scores against the episodes number
    :param scores: array including the values of the average scores to be plotted
    :return: plot figure handle
    """
    # plot the scores
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    return fig
