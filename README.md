# soccer-movement
Understanding the Movement of Professional and High School Soccer Players

''' Data sets'''
My project uses two data sets:  
1) Possession trajectories from STATS: I have not included the STATS data since I received it under an NDA. Therefore the code I provide  cannot be run on STATS data from this repository, but can be run for the high school data collected and created by me. 
2) My high school game video clips: I have converted 8 video clips into the STATS format (8 sequences of 46-dim position vectors), and included them in the file chs_data.pkl.

'''python3 velocity.py'''
plots distributions for player speed and player direction for both the STATS and high school data set. (This is Result 1 on the poster.) It also calculates a regression for player speed as a function of distance to the ball. (This is for deriving equation for s_p in Result 3.)

'''python3 regress.py'''
calculates a regression for player direction for both the STATS and high school data set. This regression is a function of angle from player to ball, angle of ball velocity, angle to nearest defensive player and angle to nearest attacking player. (This is for deriving equations for theta_p in Result 3.)

'''python3 stats_speed.py'''
calculates possession properties for the STATS data depending on whether or not a goal was scored. (This is for Result 2 of the poster.) This code is for STATS only. Student possessions pingpong between teams frequently which makes it hard to create single-possession video clips. Therefore, my student data is not for the study of possessions. 


 
