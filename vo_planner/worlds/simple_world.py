import matplotlib
#matplotlib.use('TkAgg')
import os
from copy import deepcopy
from subprocess import Popen
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
import logging
import math
from imageio import imwrite
import shutil
import numpy as np
import time
from glob import glob
import os
from imageio import imread, imwrite
from PIL import Image

goal_pixel = 255
obstacle_pixel = 50
robot_pixel = 100
true_robot_pixel = 150
min_pixel = min(goal_pixel, obstacle_pixel, robot_pixel, true_robot_pixel)
max_pixel = min(goal_pixel, obstacle_pixel, robot_pixel, true_robot_pixel)







class Particle():
    def __init__(self, world, name, local_map, init_y, init_x,
                 angle, speed, clear_map=False,
                 bounce=True, bounce_angle=45, entire_body_outside=True,
                 color=12, ymarkersize=3, xmarkersize=3):

        # if object is bouncy - markersize offsets must be possitive!
        if bounce:
            assert(ymarkersize>0)
            assert(xmarkersize>0)
        self.clear_map = clear_map
        self.entire_body_outside = entire_body_outside
        self.bounce_angle=bounce_angle
        self.steps = 0
        self.world = world
        self.name = name
        self.angle = angle
        self.speed = speed
        self.color = color
        self.bounce = bounce
        # always add positive number to size
        self.ymarkersize = abs(ymarkersize)
        self.xmarkersize = abs(xmarkersize)
        self.y = init_y
        self.x = init_x
        self.alive = True
        self.local_map = local_map
        self.step(0)

    def wall_bounce(self, hit_wall):
        if hit_wall:
            # some agents will bounce off of the walls, others should die
            if self.bounce:
                self.angle+=np.sign(self.angle)*self.bounce_angle
            else:
                self.alive = False

    def step(self, timestep):
        # (meters/second) * second
        # TODO angle
        rads = np.deg2rad(self.angle)
        #if self.name == 'robot':
        #    print("robot step", self.y, self.x, self.speed)
        self.y = self.y + self.speed*np.sin(rads)*timestep
        self.x = self.x + self.speed*np.cos(rads)*timestep
        self.plot(self.y, self.x)
        return self.alive

    def set_state(self,y,x):
        self.y = y
        self.x = x
        self.plot(self.y, self.x)
        return self.alive

    def plot(self,newy,newx):
        if self.clear_map:
            self.local_map*=0

        hit_wall = False
        # markersize is always positive so only need to check x/ysize
        newy = int(np.rint(newy))
        newx = int(np.rint(newx))
        newyplus = int(np.rint(newy+self.ymarkersize))
        newxplus = int(np.rint(newx+self.xmarkersize))
        if not self.entire_body_outside:
            # subtract one because of the way range works
            #if ((newyplus-1 < 0)  or (newxplus-1 < 0) or
            #    (newy < 0) or (newx < 0) or
            #    (newyplus-1 > self.world.ysize-1)  or
            #    (newxplus-1 > self.world.xsize-1) or
            #    (newy > self.world.ysize-1)  or
            #    (newx > self.world.xsize-1)):
            #    #print('robot bounce',newy, newx, newyplus, newxplus)

            # only bounce if all are outside of the bounds
            hit = False
            if (newy < 0):
                # hitting bottom
                newy = 0
                newyplus = int(np.rint(newy+self.ymarkersize))
                hit = True

            if (newx < 0):
                # this is hitting left wall
                newx = 0
                newxplus = int(np.rint(newx+self.xmarkersize))
                hit = True

            if (newyplus > self.world.ysize-1):
                # top wall hit
                newy = (self.world.ysize-1)-self.ymarkersize
                newyplus = int(np.rint(newy+self.ymarkersize))
                hit = True

            if (newxplus > self.world.xsize-1):
                # right size wall hit
                newx = (self.world.xsize-1)-self.xmarkersize
                newxplus = int(np.rint(newx+self.xmarkersize))
                hit = True

            self.wall_bounce(hit)

        else:
            if (((newyplus < 0)  and (newy < 0)) or
                ((newyplus > self.world.ysize-1) and (newy > self.world.ysize-1)) or
               ((newxplus < 0) and (newx < 0)) or
               ((newxplus > self.world.xsize-1) and (newx > self.world.xsize-1))):
                self.wall_bounce(True)

            # make edges within border
            if (newyplus>= self.world.ysize-1):
                newyplus = self.world.ysize-1
            if (newxplus>= self.world.xsize-1):
                newxplus = self.world.xsize-1
            if (newy>=self.world.ysize-1):
                newy = self.world.ysize-1
            if (newx>=self.world.xsize-1):
                newx = self.world.xsize-1
            if (newyplus<= 0):
                newyplus = 0
            if (newxplus<= 0):
                newxplus = 0
            if (newy <= 0):
                newy = 0
            if (newx <= 0):
                newx = 0

        if self.alive:
            y = range(newy, newyplus)
            x = range(newx, newxplus)
            if not len(y):
                y = [int(newy)]
            if not len(x):
                x = [int(newx)]
            inds = np.array([(yy,xx) for yy in y for xx in x]).T

            try:
                self.local_map[inds[0,:], inds[1,:]] = self.color
            except Exception, e:
                print('particle plot', e)
                embed()
            self.steps +=1

class SimpleEnv():
    def __init__(self, random_state, ysize, xsize, obstacle_types="NONE",
                 timestep=1,level=1, num_angles=3, agent_max_speed=1.0):
        # TODO - what if episode already exists in savedir
        self.rdn = random_state
        if obstacle_types == "NONE":
            self.obstacle_types = {
                'tree':{'color':obstacle_pixel, 'speed':[0.0],
                        'xsize':np.linspace(1, min(5,xsize),5, dtype=np.int),
                        'ysize':np.linspace(1, min(5,ysize),5, dtype=np.int),
                        'angles':[0.],},
                'fence':{'color':obstacle_pixel,
                         'speed':[0.0],
                         'xsize':np.linspace( min(30,xsize)//2, min(30,ysize),5, dtype=np.int),
                         'ysize':np.linspace(min(3,xsize)//2, min(3,xsize), 5, dtype=np.int),
                         'angles':[0.]},
                'building':{'color':obstacle_pixel, 'speed':[0.0],
                       'xsize':np.linspace(min(25,xsize)//2, min(25,xsize), 5, dtype=np.int),
                       'ysize':np.linspace(min(30,xsize)//2, min(30,xsize), 5, dtype=np.int),
                       'angles':[0.],},
                }

        else:
            self.obstacle_types = obstacle_types

        obstacle_max = max([max(o['speed']) for  n,o in self.obstacle_types.iteritems()])
        if obstacle_max > 0:
            self.dynamic = True
        else:
            self.dynamic = False
        self.max_obstacles =level
        self.steps = 0
        self.ysize = ysize
        self.xsize = xsize
        self.experiment_name = "None"

        self.timestep = timestep
        self.max_speed = agent_max_speed
        # average speed
        # make max steps twice the steps required to cross diagonally across the road
        self.max_steps = int(3*(np.sqrt(self.ysize**2 + self.xsize**2)/float(self.max_speed))/float(self.timestep))
        self.lose_reward = -20
        self.win_reward = np.abs(self.lose_reward)
        #      90
        #      |
        # 180 --- 0
        #      |
        #     270

        #self.angles = np.linspace(0, 180, 5)[::-1]
        #self.speeds = np.linspace(.1,self.max_speed,3)
        #self.angles = np.linspace(-180, 180, num_angles, endpoint=False)
        self.action_angles = np.linspace(135, 45, num_angles, endpoint=True)
        self.speeds = [self.max_speed]
        self.actions = []
        for s in self.speeds:
            for a in self.action_angles:
                self.actions.append((s,a))
        self.action_space = range(len(self.actions))
        self.road_map = np.zeros((self.ysize, self.xsize), np.uint8)

    def get_lose_reward(self, state_index):
        # lose reward is negative make step reward positive
        #return self.lose_reward
        return self.lose_reward #+ self.get_step_bonus(state_index)

    def get_step_bonus(self, state_index):
        #print("step reward", state_index, self.max_steps, sr)
        return (self.lose_reward/2.0)*(state_index/float(self.max_steps))

    def get_win_reward(self, state_index):
        print('win reward', state_index, self.win_reward + self.get_step_penalty(state_index))
        return self.win_reward + self.get_step_penalty(state_index)

    def get_step_penalty(self, state_index):
        #print("step reward", state_index, self.max_steps, sr)
        return -(self.win_reward/2.0)*(state_index/float(self.max_steps))

    def get_goal_from_roadmap(self, roadmap):
        if roadmap.max() == self.goal.color:
            return True, np.where(roadmap == self.goal.color)
        else:
            return False, ''

    def get_goal_from_state(self, state):
        goal_loc = np.where(state[1] == self.goal.color)
        return goal_loc

    def check_state(self, state, robot_is_alive, state_index):
        #assert(state[1].max() == max_pixel)
        if state_index > self.max_steps-2:
            #return True, self.get_step_penalty(state_index)
            # this makes it like the experiments that were run on the static one
            return True, 0.0 #self.get_step_penalty(state_index)
        elif not robot_is_alive:
            return True, self.get_lose_reward(state_index)
        else:
            # check for collisions
            ry, rx = self.get_robot_state(state)
            rmap = deepcopy(state[1])
            goal_loc = self.get_goal_from_state(state)
            rmap[goal_loc] -= self.goal.color
            # if particle is able to collide with obstcles
            #if rmap[ry,rx].sum()>0:
            #    return True, self.get_lose_reward(state_index)
            #elif self.goal_maps[state_index][ry,rx]>0:
            if (ry in goal_loc[0]) and (rx in goal_loc[1]):
                wreward = self.get_win_reward(state_index)
                #print("MADE IT TO GOAL", wreward, state_index)
                return True,wreward
            else:
                return False, 0.0


    def get_robot_state(self,state):
        ry = int(np.rint(state[0][0]*self.ysize))
        rx = int(np.rint(state[0][1]*self.xsize))
        if ry>self.ysize-1:
            ry = self.ysize-1
        if rx>self.xsize-1:
            rx = self.xsize-1
        if rx<0:
            rx = 0
        if ry<0:
            ry = 0
        return (ry,rx)

    def get_goal_bearing(self,state):
        gy,gx = self.get_goal_state(state)
        ry,rx = self.get_robot_state(state)
        dy = gy-ry
        dx = gx-rx
        goal_angle = np.rad2deg(math.atan2(dy,dx))
        return goal_angle

    def get_data_from_fig(self):
        data = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return data



    def create_goal(self, goal_distance):
        print("Creating goal")
        goal_ymin = max([self.robot.y-goal_distance, 2])
        goal_ymax = min([self.robot.y+goal_distance, self.ysize-2])
        assert goal_ymin > 0
        assert goal_ymax < self.ysize
        goal_xmin = max([self.robot.x-goal_distance, 2])
        goal_xmax = min([self.robot.x+goal_distance, self.xsize-2])
        assert goal_xmin > 0
        assert goal_xmax < self.xsize

        goal_y = float(self.rdn.randint(goal_ymin,goal_ymax))
        goal_x = float(self.rdn.randint(goal_xmin,goal_xmax))
        self.goal_maps = np.zeros((self.max_steps, self.ysize, self.xsize), np.uint8)
        self.road_maps = np.zeros((self.max_steps, self.ysize, self.xsize), np.uint8)
        goal_angle = self.rdn.choice(range(1, 359, 45))
        self.goal = Particle(world=self, name='goal',
                              local_map=self.goal_maps[0],
                              init_y=goal_y, init_x=goal_x,
                              angle=goal_angle, speed=self.goal_speed,
                              clear_map=True,
                              bounce=True, entire_body_outside=False,
                              ymarkersize=2, xmarkersize=2,
                              color=goal_pixel)

    def reset(self, goal_distance=1000, experiment_name="None", condition_length=0, goal_speed=0.5):

        if goal_speed != 0.5:
            print("WARNING GOAL SPEED ISNT 0.5")
            embed()
        self.goal_speed = goal_speed
        self.experiment_name = experiment_name
        max_xobstaclesize = int(self.xsize*.15)

        self.plotted = False
        plt.close()

        # robot shape
        yrsize,xrsize=1,1
        self.safezone = yrsize*1
        init_ys = [0, self.ysize-(1+yrsize)]
        init_y = float(self.rdn.choice(init_ys))
        init_x = float(self.rdn.randint(xrsize+5,self.xsize-(xrsize+5)))
        self.robot_map = np.zeros((self.ysize, self.xsize), np.uint8)
        self.true_robot_map = np.zeros((self.ysize, self.xsize), np.uint8)

        self.robot = Particle(world=self,  name='robot',
                              local_map=self.robot_map,
                              init_y=init_y, init_x=init_x,
                              angle=0, speed=0.0, clear_map=True,
                              bounce=False, entire_body_outside=False, # robot must not bounce
                              xmarkersize=xrsize, ymarkersize=yrsize,
                              color=robot_pixel)

        self.true_robot = Particle(world=self,  name='vo',
                              local_map=self.true_robot_map,
                              init_y=init_y, init_x=init_x,
                              angle=0, speed=0.0, clear_map=True,
                              bounce=False, entire_body_outside=False, # robot must not bounce
                              xmarkersize=xrsize, ymarkersize=yrsize,
                              color=true_robot_pixel)



        # only allow goal to be so far away
        self.create_goal(goal_distance)
        self.obstacles = {}
        self.cnt = 0
        # initialize map
        [self.step_obstacles() for i in range(self.xsize)]
        # make all of the obstacles
        self.no_goal_states = []
        self.goals = []
        for t in range(self.max_steps):
            self.step_obstacles()
            self.goal_maps[t] = self.goal.local_map
            # road map has goal and road
            self.road_maps[t] = self.goal.local_map
            nz = self.road_map>0
            self.road_maps[t,nz] = self.road_map[nz]
            self.goals.append((self.goal.y, self.goal.x))
            if self.road_maps[t].max() < self.goal.color:
                self.no_goal_states.append(t)
            #self.render([(0.4, 0.3), self.road_maps[t]])
        state = self.get_state(condition_length)
        if not self.robot.alive:
            print('robot died')
            embed()
        return state

    def step_obstacles(self):
        self.road_map *=0
        dead_obstacles = []
        self.goal.step(self.timestep)
        for n,o in self.obstacles.iteritems():
            alive = o.step(self.timestep)
            if not alive:
                dead_obstacles.append(n)

        for n in dead_obstacles:
            del self.obstacles[n]

        self.add_obstacles()

    def add_obstacles(self):
        if len(self.obstacles) < self.max_obstacles:
            olist = [obstacle for name, obstacle in self.obstacle_types.iteritems()]
            for obstacle in olist:
                if len(self.obstacles) < self.max_obstacles:
                    pp = self.rdn.poisson(min(self.ysize,self.xsize)//2)
                    xs = self.rdn.choice(obstacle['xsize'])
                    ys = self.rdn.choice(obstacle['ysize'])
                    if min(xs,ys) < pp:
                        angle = self.rdn.choice(obstacle['angles'])
                        init_x = self.rdn.choice(np.arange(1,self.xsize-1-xs))
                        init_y = self.rdn.choice(np.arange(1,self.xsize-1-xs))
                        self.obstacles[self.cnt] = Particle(world=self, name=self.cnt,
                                                     local_map=self.road_map,
                                                     init_y=init_y,
                                                     init_x=init_x,
                                                     angle=angle,
                                                     speed=self.rdn.choice(obstacle['speed']),
                                                     bounce=False,
                                                     color=obstacle['color'],
                                                     ymarkersize=ys,
                                                     xmarkersize=xs)
                        self.cnt +=1
#
#    def set_road_maps(self, road_maps):
#        self.road_maps = road_maps
#        self.max_steps = len(self.road_maps)
#

    def get_state_given_roadmap(self, road_map):
        #if road_map.max() != max_pixel:
        #    print("given map with no goal")
        rstate = (self.robot.y/float(self.ysize), self.robot.x/float(self.xsize))
        state = (rstate, road_map)
        return state

    def get_goal_state(self):
        # instantanous goal
        gstate = (self.goal.y/float(self.ysize), self.goal.x/float(self.xsize))
        return gstate

    def get_state(self, state_index):
        road_map = self.get_road_state(state_index)
        if road_map.max() != max_pixel:
            print("NO GOAL STEP ", state_index)
        rstate = (self.robot.y/float(self.ysize), self.robot.x/float(self.xsize))
        state = (rstate, road_map)
        return state

    def set_state(self, state, state_index):
        self.robot.alive = True
        robot_val = state[0]
        ry = float(robot_val[0]*self.ysize)
        rx = float(robot_val[1]*self.xsize)
        #print("want to set robot to", ry,rx)
        robot_alive = self.robot.set_state(ry,rx)
        finished, reward = self.check_state(state, robot_alive, state_index)
        return finished, reward

    def get_road_state(self, state_index):
        try:
            road_map = self.road_maps[state_index]
            return road_map
        except Exception, e:
            print("ROAD STATE TOO MANY STEPS")
            embed()

    def set_action_values_from_index(self, action_index):
        assert 0 <= action_index < len(self.action_space)
        action_key = self.actions[action_index]
        speed, angle = action_key[0], action_key[1]
        self.robot.speed = speed
        self.robot.angle = angle
        return speed, angle

    def model_step(self, state, state_index, action_index, next_road_map):
        ''' step agent '''
        finished, reward = self.set_state(state, state_index)
        if finished:
            return state, reward, finished, ''
        else:
            self.set_action_values_from_index(action_index)
            # road_maps is max_steps long
            # robot is alive will say if the robot ran into a wall
            robot_is_alive = self.robot.step(self.timestep)
            #print('##################################')
            #print('## rstep alive:{} action: {} speed: {} angle {} ({},{}) step {}'.format(robot_is_alive,
            #      action_index, self.robot.speed, self.robot.angle,
            #      round(self.robot.y,2), round(self.robot.x,2), state_index))
            #print('##################################')

            #next_state = self.get_state(next_state_index)
            #assert(next_state[1].max() == max_pixel)
            # reward for time step
            next_state_index = state_index + 1
            next_state = self.get_state_given_roadmap(next_road_map)
            finished, reward = self.check_state(next_state, robot_is_alive, next_state_index)
            return next_state, reward, finished, ''



    def step(self, state, state_index, action_index):
        ''' step agent '''
        finished, reward = self.set_state(state, state_index)
        if finished:
            return state, reward, finished, ''
        else:
            self.set_action_values_from_index(action_index)
            # road_maps is max_steps long
            # robot is alive will say if the robot ran into a wall
            robot_is_alive = self.robot.step(self.timestep)
            #print('##################################')
            #print('## rstep alive:{} action: {} speed: {} angle {} ({},{}) step {}'.format(robot_is_alive,
            #      action_index, self.robot.speed, self.robot.angle,
            #      round(self.robot.y,2), round(self.robot.x,2), state_index))
            #print('##################################')

            next_state_index = state_index + 1
            next_state = self.get_state(next_state_index)
            #assert(next_state[1].max() == max_pixel)
            # reward for time step
            finished, reward = self.check_state(next_state, robot_is_alive, next_state_index)
            return next_state, reward, finished, ''

    def close_plot(self):
        plt.clf()
        plt.close()
        try:
            del self.fig
            del self.ax
            del self.shown
        except:
            pass
        self.plotted = False

    def get_state_plot(self, state):
        try:
            self.robot.alive = True
            ry = float(state[0][0]*self.ysize)
            rx = float(state[0][1]*self.xsize)
            self.robot.set_state(ry,rx)
            #show_state = state[1]+self.robot.local_map#+self.goal.local_map
            show_state = state[1]+self.robot.local_map+self.goal.local_map
        except Exception, e:
            print("env state plot")
            embed()
        return show_state

    def render(self, state):
        show_state = self.get_state_plot(state)
        if not self.plotted:
            # reset environment
            self.plotted = True
            plt.ion()
            self.fig, self.ax = plt.subplots(1,1)
            plt.title(self.experiment_name)
            self.shown = self.ax.imshow(self.road_map, vmin=min_pixel, vmax=max_pixel, origin='lower', interpolation="none")
            self.ax.set_aspect('equal')
            self.ax.set_ylim(0,self.ysize)
            self.ax.set_xlim(0,self.xsize)

        self.shown.set_data(show_state)
        plt.show()
        plt.pause(.0001)

if __name__ == '__main__':
    # generate training data
    ysize, xsize = 200,300
    train = True
    if train:
        dirname = 'train'
        seed = 700
        num_episodes = 1
    else:
        dirname = 'test'
        seed = 40
        num_episodes = 3
    save_path = '%s_imgs_%sx%s/'%(dirname, ysize,xsize)

    if not os.path.exists(save_path):
        os.makedirs(save_path)



    rdn = np.random.RandomState(seed)
    level= 10
    env = SimpleEnv(random_state=rdn, ysize=ysize, xsize=xsize,  level=level, agent_max_speed= 0.5,)
    print(env.max_steps)
    #env = SimpleEnv(random_state=rdn, ysize=ysize, xsize=xsize, level=level, agent_max_speed= 1.0)
    #print(env.max_steps)
    #sys.exit()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for e in range(num_episodes):
        print(e/float(num_episodes))
        env.reset()
        #for t in range(env.road_maps.shape[0]):
        for t in [0]:
            name = os.path.join(save_path,'seed_%05d_episode_%05d_frame_%05d.png'%(seed, e, t))
            p = env.get_state_plot(env.get_state(0))
            plt.figure()
            plt.imshow(p,origin='lower',interpolation='none',cmap=plt.cm.viridis)
            plt.savefig(name)

            #imwrite(name,p)

    embed()

