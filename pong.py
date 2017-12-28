""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import pickle
import argparse
import gym
from datetime import datetime

# Args and hyperparams
parser = argparse.ArgumentParser(description='Presupposition attention')
add = parser.add_argument
add('--H', type=int, default=200, help='number of hidden layer neurons')
add('--batch_size', type=int, default=10,
              help='number of episodes before updating our network parameters')
add('--learning_rate', type=float, default=1e-3)
add('--gamma', type=float, default=0.99, help='discount factor for reward')
add('--decay_rate', type=float, default=0.99, help='decay factor for RMSProp leaky sum of grad^2')
add('--resume', action='store_true', default=False)
add('--render', action='store_true', default=False)
add('--gpu', action='store_true', default=False)
args = parser.parse_args()

H = args.H
batch_size = args.batch_size
learning_rate = args.learning_rate
gamma = args.gamma
decay_rate = args.decay_rate
resume = args.resume
render = args.render
D = 80 * 80 # input dimensionality: 80x80 grid

# If GPU compute, import CuPy instead of Numpy
if args.gpu:
  import cupy as xp
else:
  import numpy as xp

if resume:
  pong        = pickle.load(open('save.p', 'rb'))
  total_hours = pong['total_time']
  episodes    = pong['total_ep']
  model       = pong['model']
  print('****RESUMING TRAINING*****')
else:
  model       = {}
  model['W1'] = xp.random.randn(H,D) / xp.sqrt(D) # "Xavier" initialization
  model['W2'] = xp.random.randn(H) / xp.sqrt(H)
  total_hours = 0
  episodes    = 0

t1 = datetime.now()
# There are many episodes before updating our parameters, these accumulate
# in the gradient buffer
grad_buffer = {k : xp.zeros_like(v) for k,v in model.items() }
# rmsprop memory
rmsprop_cache = { k : xp.zeros_like(v) for k,v in model.items() }

def sigmoid(x):
  return 1.0 / (1.0 + xp.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  if args.gpu: I = xp.asarray(I)
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype('float64').ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = xp.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    # running_add is the future reward to discount. If r[t] already has a
    # reward, that means we are at the end of the game, there is no future
    # state to discount, so running_add = 0
    if r[t] != 0: running_add = 0
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add

  return discounted_r

def policy_forward(x):
  h = xp.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity, wow!!
  logp = xp.dot(model['W2'], h) # logp is a scalar
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):

  """
  backward pass. (eph is array of intermediate hidden states)
  Args:
    eph: episode hidden states, stacked
    epdlogp: episode action gradient with advantage
  """
  dW2 = xp.dot(eph.T, epdlogp).ravel() # derivs wrt W2
  dh = xp.outer(epdlogp, model['W2']) # derivs wrt h post activation
  dh[eph <= 0] = 0 # backprop relu
  dW1 = xp.dot(dh.T, epx) # derivs wrt W1
  return {'W1':dW1, 'W2':dW2}

env = gym.make("Pong-v0")
observation = env.reset() # start the game
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[] # x, h state, deriv predict, rewards
running_reward = None
reward_sum = 0
episode_number = 0
while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else xp.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = 2 if xp.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to
  # be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
  # Later, we multiply this gradient by the reward. So if we take action 1, but
  # lets say 'aprob' is quite, small, then the gradient is large. If reward for
  # action 1 is positive, then we encourage our network to increase aprob

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished, accumulate gradients or backprop
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = xp.vstack(xs)
    eph = xp.vstack(hs)
    epdlogp = xp.vstack(dlogps) # episode gradients of y - aprob
    # epr = xp.vstack(drs) # this won't work with cupy
    epr = xp.array(drs)
    epr = xp.expand_dims(epr, axis=1) # equivalent to vstack
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the grad var)
    discounted_epr -= xp.mean(discounted_epr)
    discounted_epr /= xp.std(discounted_epr)

    # Modulate the gradient with advantage (PG magic happens right here.)
    epdlogp *= discounted_epr
    grad = policy_backward(eph, epdlogp)
    # Gradients are accumulated over many episodes, determined by batch size
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    # RMSprop: divide the gradient by a running average of its recent magnitude
    # see 6e, slide 26 of http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    # for other opt, see: http://ruder.io/optimizing-gradient-descent/index.html#rmsprop
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient for weight k
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (xp.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = xp.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    if running_reward is None:
      running_reward = reward_sum
    else:
      running_reward = running_reward * 0.99 + reward_sum * 0.01

    # Print some info
    print('resetting env. episode reward total was {}. running mean: {}'.format(\
                                                  reward_sum, running_reward))
    t2 = datetime.now()
    hours = (t2 - t1).total_seconds()/60/60
    training_time = total_hours + hours
    total_ep = episodes + episode_number
    print('total training hours: {}'.format(training_time))
    print('total episodes: {}'.format(total_ep))

    # Maybe save
    if episode_number % 100 == 0:
      pong = {'total_time': training_time, 'model':model, 'total_ep':total_ep}
      pickle.dump(pong, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    end = '' if reward == -1 else '!!!!!!'
    print('ep {}: game finished, reward: {}{}'.format(episode_number,reward,end))



