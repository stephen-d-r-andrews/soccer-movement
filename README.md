# soccer-movement
Understanding the Movement of Professional and High School Soccer Players

python3 velocity.py plots distributions for player speed and player direction for both the STATS and high school data set. It also calculates a regression for player speed as a function of distance to the ball. 

python3 regress.py calculates a regression for player direction for both the STATS and high school data set. This regression is a function of angle from player to to ball, angle of ball velocity, angle to nearest defensive player and angle to nearest attacking player.

python3 stats_speed.py calculates sequence properties for the STATS data depending on whether or not a goal was scored.

I have included the high school data in the file chs_data.pkl. I have not included the STATS data since I received it under and NDA. As a result the code in stats_speed.py cannot be run from this repository.

