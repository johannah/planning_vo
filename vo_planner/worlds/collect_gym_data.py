#import matplotlib
#matplotlib.use("Agg")
import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import time
from copy import copy
import os

NUM_EPISODES = 20
save_data_file = 'human_road'

if __name__ == '__main__':
    from pyglet.window import key

    a = np.array( [0.0, 0.0] )
    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key._6:  a[0] = -1.0
        if k==key._7:  a[0] = -0.5
        if k==key._8:  a[0] = 0.5
        if k==key._9:  a[0] = 1.0 
        if k==key._1:  a[1] = +.2
        if k==key._2:  a[1] = +.4
        if k==key._3:  a[1] = +.6
        if k==key._4:  a[1] = +.8
        if k==key.DOWN:  a[1] = -.5  # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k==key._6 and a[0] == -1.0: a[0]=0
        if k==key._7 and a[0] == -0.5: a[0]=0
        if k==key._8 and a[0] == 0.5: a[0]=0
        if k==key._9 and a[0] == 1.0: a[0]=0

        if k in [key._1, key._2, key._3, key._4]: a[1]=0
        if k==key.DOWN:  a[1] = 0
    env = gym.make('CarRacing-v0')
    env.mode = "human"
    env.reset()

    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release
    all_states = []
    all_actions = []
    all_roads = []
    for xxx in range(NUM_EPISODES):
        track = env.reset()
        all_roads.append((track))
        total_reward = 0.0
        steps = 0
        restart = False
        states = []
        actions = []
        cnt = 0
        while True:
            # a is (steer, gas, brake)
            take_action = copy(a)
            print(take_action)
            s, r, done, info = env.step(take_action)
            #if take_action.sum()>0:
            #print('action', cnt, take_action)
            cnt+=1
            total_reward += r
            actions.append(take_action)
            states.append(s)
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            if not record_video: # Faster, but you can as well call env.render() every time to play full window.
                env.render()
            if done or restart: 
                all_states.append(states)
                all_actions.append(actions)
                break
    env.close()
    if os.path.exists(save_data_file+'.npz'):
        data = np.load(save_data_file+'.npz')
        all_states.extend(data['states'])
        all_roads.extend(data['actions'])
        all_actions.extend(data['actions'])
    np.savez(save_data_file, states=all_states, roads=all_roads, actions=all_actions)
