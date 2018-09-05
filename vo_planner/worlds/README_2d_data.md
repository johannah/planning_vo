The numpy arrays contain three data products:
'states':
 [episode, timestep, features], where features are: 
 [posy,posx,speed,angle,steer,gas,brake,dify,diffx,steering_after,throttle_after]
'details': indicates information about how these examples were extracted
[index_of_episode, start_time_index, end_time_index, length_in_time]
'subgoals': indexes of subgoals to use for training examples. These are the indexes to [posy, posx] which reward should be evaluated on. 
