import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
import tensorflow as tf
import pickle
import random
import keras
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam, SGD

# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this is a little helper function that calculates the Q error for you
# so that you can easily use it in tensorflow as the loss
# you can copy this into your agent class or use it from here
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def Q_loss(Q_s, action_onehot, Q_s_next, best_action_next, reward, terminal, discount=0.99):
    """
    All inputs should be tensorflow variables!
    We use the following notation:
       N : minibatch size
       A : number of actions
    Required inputs:
       Q_s: a NxA matrix containing the Q values for each action in the sampled states.
            This should be the output of your neural network.
            We assume that the network implments a function from the state and outputs the 
            Q value for each action, each output thus is Q(s,a) for one action 
            (this is easier to implement than adding the action as an additional input to your network)
       action_onehot: a NxA matrix with the one_hot encoded action that was selected in the state
                      (e.g. each row contains only one 1)
       Q_s_next: a NxA matrix containing the Q values for the next states.
       best_action_next: a NxA matrix with the best current action for the next state
       reward: a Nx1 matrix containing the reward for the transition
       terminal: a Nx1 matrix indicating whether the next state was a terminal state
       discount: the discount factor
    """
    # calculate: reward + discount * Q(s', a*),
    # where a* = arg max_a Q(s', a) is the best action for s' (the next state)
    target_q = (1. - terminal) * discount * tf.reduce_sum(best_action_next * Q_s_next, 1, keep_dims=True) + reward
    # NOTE: we insert a stop_gradient() operation since we don't want to change Q_s_next, we only
    #       use it as the target for Q_s
    target_q = tf.stop_gradient(target_q)
    # calculate: Q(s, a) where a is simply the action taken to get from s to s'
    selected_q = tf.reduce_sum(action_onehot * Q_s, 1, keep_dims=True)
    target_q = tf.cast(target_q, tf.float64)
    selected_q = tf.cast(selected_q, tf.float64)
    loss = tf.reduce_sum(tf.square(selected_q - target_q))
    #loss = tf.Session().run(loss)
    return loss

def append_to_hist(state, obs):
    """
    Add observation to the state.
    """
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# In contrast to your last exercise you DO NOT generate data before training
# instead the TransitionTable is build up while you are training to make sure
# that you get some data that corresponds roughly to the current policy
# of your agent
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)

if opt.disp_on:
    win_all = None
    win_pob = None


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# You should prepare your network training here. I suggest to put this into a
# class by itself but in general what you want to do is roughly the following
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
# setup placeholders for states (x) actions (u) and rewards and terminal values
x = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
u = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
ustar = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.act_num))
xn = tf.placeholder(tf.float32, shape=(opt.minibatch_size, opt.hist_len*opt.state_siz))
r = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))
term = tf.placeholder(tf.float32, shape=(opt.minibatch_size, 1))

conv1_weights = tf.Variable(
  tf.truncated_normal([5, 5, 1, 16],  # 5x5 filter, depth 16.
                      stddev=0.1))
conv1_biases = tf.Variable(tf.zeros([16]))
conv2_weights = tf.Variable(
  tf.truncated_normal([3, 3, 16, 32], # 3x3 filter, depth 32
                      stddev=0.1))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[32]))

conv3_weights = tf.Variable(
  tf.truncated_normal([2, 2, 32, 64], # 3x3 filter, depth 64
                      stddev=0.1))
conv3_biases = tf.Variable(tf.constant(0.1, shape=[64]))

fc1_weights = tf.Variable(  # fully connected, depth 128.
  tf.truncated_normal([256, 512],
                      stddev=0.1))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
fc2_weights = tf.Variable(
  tf.truncated_normal([512, 4],
                      stddev=0.1))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[4]))

# get the output from your network
Q = my_network_forward_pass(x)
Qn =  my_network_forward_pass(xn)

# calculate the loss
loss = Q_loss(Q, u, Qn, ustar, r, term)

# setup an optimizer in tensorflow to minimize the loss
"""

# Input Dimensions
ROWS = 30
COLS = 30
CHANNELS_NUM = 4

# Q Model
def model_q():
    model = Sequential()
    model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same', input_shape=(ROWS, COLS, CHANNELS_NUM)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(opt.act_num))

    adam = Adam(lr=1e-3)
    model.compile(loss='mse',optimizer=adam)

    return model

# Target model
def model_target():
    model = Sequential()
    model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same', input_shape=(ROWS, COLS, CHANNELS_NUM)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(opt.act_num))

    adam = Adam(lr=1e-3)
    model.compile(loss='mse',optimizer=adam)
    return model


# initialize models
model_q = model_q()
model_target = model_target()

# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it
steps = 1 * 10**6
#steps = 51000
epi_step = 0
nepisodes = 0

# variables
explore = 50000  # explore randomly
decay_factor = 0.999982 # decay factor
epsilon_start = 1
epsilon_final = 0.1
epsilon = epsilon_start
epsilon_stop_logging = 0
loss = 0
discount = 0.99
flagt = False # terminal state
s = "" #action source
steps_test = 5000
st = [] #steps
ll = [] #loss
re = [] #Rewards
q_values = []#Q-Values
epsilon_decay = []
reward_sum = 0

# variables for plotting
average_episodes_lengths = []
average_episodes_rewards = []
average_episodes_successed = []
average_episodes_failed = []

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

def test_agent():
    episodes_test = 50
    successed = 0
    failed = 0
    episode_length = []
    episode_reward = []
    print("testing model")
    state = sim.newGame(opt.tgt_y, opt.tgt_x)
    test_state_with_history = np.zeros((opt.hist_len, opt.state_siz))
    append_to_hist(test_state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
    test_next_state_with_history = np.copy(test_state_with_history)
    for episode in range(episodes_test):
        test_step = 0
        test_reward = 0
        while(1):
            if state.terminal or test_step >= opt.early_stop:
                if(state.terminal):
                    successed += 1
                else:
                    failed += 1
                # reset the game
                state = sim.newGame(opt.tgt_y, opt.tgt_x)
                # reset the history
                test_state_with_history[:] = 0
                append_to_hist(test_state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
                test_next_state_with_history = np.copy(test_state_with_history)
                episode_length.append(test_step)
                episode_reward.append(test_reward)
                print("completed 1 episode with steps = ", test_step, " and reward = ", test_reward)
                break
            test_step += 1
            action = np.argmax(model_q.predict((test_state_with_history).reshape(1,30,30,4)))
            action_onehot = trans.one_hot_action(action)
            #Take next step according to the action selected
            next_state = sim.step(action)
            # append state to history
            append_to_hist(test_next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))

            # next state as current state
            test_state_with_history = np.copy(test_next_state_with_history)
            state = next_state
            test_reward += state.reward
            print("step", test_step, "reward", test_reward)
            #opt.disp_on = True
            #if opt.disp_on:
            #    win_all = None
            #    win_pob = None
            if opt.disp_on:
                if win_all is None:
                    plt.subplot(121)
                    win_all = plt.imshow(state.screen)
                    plt.subplot(122)
                    win_pob = plt.imshow(state.pob)
                else:
                    win_all.set_data(state.screen)
                    win_pob.set_data(state.pob)
                plt.pause(opt.disp_interval)
                plt.draw()
    average_episodes_lengths.append(np.mean(episode_length))
    average_episodes_rewards.append(np.mean(episode_reward))
    average_episodes_successed.append(successed/episodes_test)
    average_episodes_failed.append(failed/episodes_test)

# Training / Testing
Training = False # Variable used for testing or training
if(Training):
    for step in xrange(1, steps+1):
        if state.terminal or epi_step >= opt.early_stop:
            epi_step = 0
            nepisodes += 1
            # reset the game
            state = sim.newGame(opt.tgt_y, opt.tgt_x)
            # reset the history
            state_with_history[:] = 0
            append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
            next_state_with_history = np.copy(state_with_history)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: here you would let your agent take its action
        #       remember
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # this just gets a random action
        print("step: ", step)
        epi_step += 1
        s = np.random.choice(['random', 'network'], p=[epsilon, 1-epsilon])
        print("I have now to choose action", s)
        if (s == "random") or (step < decay_factor):
            action = randrange(opt.act_num)
            s = "Random"
            print("Random")
        else:
            reshaped_hist = state_with_history.reshape(1, ROWS, COLS, CHANNELS_NUM)
            action = model_q.predict(reshaped_hist)
            action = np.argmax(action)
            print("network")
            s="Network"
        action_onehot = trans.one_hot_action(action)
        next_state = sim.step(action)
        # append to history
        append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
        # add to the transition table
        trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)
        # mark next state as current state
        state_with_history = np.copy(next_state_with_history)
        state = next_state
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: here you would train your agent
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if step > explore:
            state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
            # TODO train me here
            # this should proceed as follows:
            # 1) pre-define variables and networks as outlined above
            # 1) here: calculate best action for next_state_batch
            # TODO:
            # action_batch_next = CALCULATE_ME
            # 2) with that action make an update to the q values
            #    as an example this is how you could print the loss 
            #print(sess.run(loss, feed_dict = {x : state_batch, u : action_batch, ustar : action_batch_next, xn : next_state_batch, r : reward_batch, term : terminal_batch}))
            action_batch_next = model_target.predict_on_batch(next_state_batch.reshape(opt.minibatch_size, ROWS, COLS, CHANNELS_NUM))
            action_batch_next_max = np.amax(action_batch_next, axis=1)
            action_batch_next_max_reshape = action_batch_next_max.reshape(opt.minibatch_size, 1)
            updated_Q = model_q.predict_on_batch(state_batch.reshape(opt.minibatch_size, ROWS, COLS, CHANNELS_NUM))
            
            # Compute model loss
            #Q_loss(Q_s, action_onehot, Q_s_next, best_action_next, reward, terminal, discount=0.99)
            loss = Q_loss(updated_Q, action_onehot, action_batch_next, action_batch_next_max_reshape, state.reward, state.terminal, discount=0.99)
            reward_sum += state.reward
            # every 100 steps
            if(step % 100) == 0:
                flagt= False
                st.append(step)
                ll.append(loss)
                re.append(reward_sum)
                q_values.append(np.mean(updated_Q))
            # Update Target model every 10000 steps
            if (step % 10000) == 0:
                print('Target model\'s weights updated')
                model_target.set_weights(model_q.get_weights())

            if (step % 1000) == 0:
                test_agent()

            # Decay epsilon among 130,000 steps until 0.1
            if (epsilon > epsilon_final):
                epsilon = epsilon*decay_factor
            else:
                epsilon = epsilon_final
                epsilon_stop_logging += 1

            if epsilon_stop_logging <= 10000:
                epsilon_decay.append(epsilon)

        # TODO every once in a while you should test your agent here so that you can track its performance
        opt.disp_on = False
        if opt.disp_on:
            if win_all is None:
                plt.subplot(121)
                win_all = plt.imshow(state.screen)
                plt.subplot(122)
                win_pob = plt.imshow(state.pob)
            else:
                win_all.set_data(state.screen)
                win_pob.set_data(state.pob)
            plt.pause(opt.disp_interval)
            plt.draw()

        # save model
        if step == 1000000:
            model_json = model_q.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model_q.save_weights("model.h5")
            print("Saved model to disk")
    
    # run session on loss tensors       
    ll = tf.Session().run(ll)

    # save results logs
    np.savetxt('Test_Episodes_Lengths.txt', average_episodes_lengths, fmt="%.5f")
    np.savetxt('Test_Episodes_Rewards.txt', average_episodes_rewards, fmt="%.5f")
    np.savetxt('Test_Episodes_successed.txt', average_episodes_successed, fmt="%.5f")
    np.savetxt('Test_Episodes_Failed.txt', average_episodes_failed, fmt="%.5f")
    np.savetxt('Loss.txt', ll, fmt="%.5f")
    np.savetxt('Reward Sum (Training).txt', re, fmt="%.5f")
    np.savetxt('Q_Values (Training).txt', q_values, fmt="%.5f")

    # summarize history for episodes lengths
    fig1 = plt.figure()
    plt.plot(average_episodes_lengths)
    plt.title('Average Episodes Lengths (50)')
    plt.ylabel('average length')
    plt.xlabel('Test Steps')
    #plt.show()
    fig1.savefig('episodes_lengths.png')

    # summarize history for episodes rewards
    fig2 = plt.figure()
    plt.plot(average_episodes_rewards)
    plt.title('Average Episodes Rewards (50)')
    plt.ylabel('average reward')
    plt.xlabel('Test Steps')
    #plt.show()
    fig2.savefig('episodes_rewards.png')

    # summarize history for loss
    fig3 = plt.figure()
    plt.plot(ll)
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('steps')
    #plt.show()
    fig3.savefig('episodes_loss.png')

    # summarize history for episodes rewards
    fig4 = plt.figure()
    plt.plot(re)
    plt.title('Reward Sum (Training)')
    plt.ylabel('reward sum')
    plt.xlabel('steps')
    #plt.show()
    fig4.savefig('reward_sum.png')

    # summarize history for epsilon decays
    fig5 = plt.figure()
    plt.plot(epsilon_decay)
    plt.title('Epsilon Decay')
    plt.ylabel('epsilon')
    plt.xlabel('steps')
    #plt.show()
    fig5.savefig('epsilon_decay.png')

    # summarize history for terminal episodes
    fig6 = plt.figure()
    plt.plot(average_episodes_successed)
    plt.title('Average Episodes successed')
    plt.ylabel('No of Episodes')
    plt.xlabel('Test Steps')
    #plt.show()
    fig6.savefig('episodes_successed.png')

    # summarize history for terminal episodes
    fig7 = plt.figure()
    plt.plot(average_episodes_failed)
    plt.title('Average Episodes Failed')
    plt.ylabel('No of Episodes')
    plt.xlabel('Test Steps')
    #plt.show()
    fig7.savefig('episodes_failed.png')

    # summarize history for Q Values
    fig8 = plt.figure()
    plt.plot(q_values)
    plt.title('Q-Values')
    plt.ylabel('Q-Values')
    plt.xlabel('steps')
    #plt.show()
    fig8.savefig('q_values.png')

# 2. perform a final test of your model and save it
# TODO
else:
    successed = 0
    failed = 0
    json_file = open('model.json','r')
    modell = json_file.read()
    json_file.close()
    ml = model_from_json(modell)
    ml.load_weights("model.h5")
    print("Model loaded")
    state = sim.newGame(opt.tgt_y, opt.tgt_x)
    steps_test = 5000
    state_with_history = np.zeros((opt.hist_len, opt.state_siz))
    append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
    next_state_with_history = np.copy(state_with_history)
    for step in range(steps_test):
        if state.terminal or epi_step >= opt.early_stop:
            epi_step = 0
            nepisodes += 1
            if(state.terminal):
                successed += 1
            else:
                failed += 1
            # reset the game
            state = sim.newGame(opt.tgt_y, opt.tgt_x)
            # and reset the history
            state_with_history[:] = 0
            append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
            next_state_with_history = np.copy(state_with_history)
        epi_step += 1
        action = np.argmax(ml.predict((state_with_history).reshape(1,30,30,4)))
        action_onehot = trans.one_hot_action(action)
        #Take next step according to the action selected
        next_state = sim.step(action)
        # append state to history
        append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))

        # mark next state as current state
        state_with_history = np.copy(next_state_with_history)
        state = next_state
        if opt.disp_on:
            if win_all is None:
                plt.subplot(121)
                win_all = plt.imshow(state.screen)
                plt.subplot(122)
                win_pob = plt.imshow(state.pob)
            else:
                win_all.set_data(state.screen)
                win_pob.set_data(state.pob)
            plt.pause(opt.disp_interval)

