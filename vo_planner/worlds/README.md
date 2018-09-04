A group of worlds for testing the racecar data. 

This is dependendent on a version of the car_racing game in gym. To run
clone:

git clone http://github.com/johannah/gym

Update to the "racecar" branch. 

To collect human driving data, run: 
`python collect_gym_data.py` 
The human should see a car racing environment pop up. Drive the car along the road using the following commands: 
Steering: 
far left=6, slight left=7, slight right=8, right=9

Speed (larger is faster): 1,2,3,4

Brake: down arrow

Extract collected data by running `python extract_data.py`. 




