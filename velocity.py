import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy.linalg import norm

dataset = 'chs' # 'stats' or 'chs'
if dataset=='stats':
    rawdata=pickle.load(open('train_data.pkl','rb'),encoding='latin1')
    stepsize=1
    seq2process=7500
    skip=100
else:
    rawdata=pickle.load(open('chs_data.pkl','rb'),encoding='latin1')
    stepsize=10
    seq2process=8
    skip=1

balldist = []
playerspeed = []
player_relangle = []
player_balldirangle = []
player_angle = []
ballanglestore = []

for i in range(0,seq2process):    
    key = 'sequence_'+str(i+1)
    print(key)
    seq = rawdata[key]
    x_range = np.shape(seq)[0]-stepsize
    for t in range(x_range):
        def legit(t,x):
            if dataset=='stats':
                return True
            if min(seq[t,x],seq[t,x+1])<-40.:
                return False
            return True

        ballvel = tuple([(10./stepsize)*elem for elem in (seq[t+stepsize,44]-seq[t,44],seq[t+stepsize,45]-seq[t,45])])
        ballspeed = norm(ballvel)

        for p in range(0,44,2):
            if legit(t,44) and legit(t+stepsize,44) and legit(t,p) and legit(t+stepsize,p):
                vel = tuple([(10./stepsize)*elem for elem in (seq[t+stepsize,p]-seq[t,p],seq[t+stepsize,p+1]-seq[t,p+1])])
                z = (seq[t,44]-seq[t,p])+1j*(seq[t,45]-seq[t,p+1])            
                ballangle = np.angle(z)
                ballanglestore.append([ballangle])
                balldist.append([np.abs(z)])
                z = vel[0]+1j*vel[1]
                if np.abs(z)>6.:
                    print(key,t,p)
                playerspeed.append([np.abs(z)])
                velangle = np.angle(z)
                player_angle.append([velangle])
                
                relangle = velangle-ballangle
                if relangle>np.pi:
                    relangle -= np.pi
                elif relangle<-np.pi:
                    relangle += np.pi
                player_relangle.append(relangle)

                balldirangle = velangle-np.angle(ballvel[0]+1j*ballvel[1])
                if balldirangle>np.pi:
                    balldirangle -= np.pi
                elif balldirangle<-np.pi:
                    balldirangle += np.pi
                player_balldirangle.append(balldirangle)

balldist = np.array(balldist)
playerspeed = np.array(playerspeed)
player_relangle = np.array(player_relangle)
player_balldirangle = np.array(player_balldirangle)
player_angle = np.array(player_angle)
ballanglestore = np.array(ballanglestore)
speed_reg = LinearRegression().fit(balldist, playerspeed)
print('speed coef',speed_reg.coef_)
print('speed intercept',speed_reg.intercept_)
angle_reg = LinearRegression().fit(ballanglestore, player_angle)
print('angle coef',angle_reg.coef_)
print('angle intercept',angle_reg.intercept_)

if dataset == 'stats':
    prefix = ''
else:
    prefix = dataset+'-'

f = plt.figure(1)
plt.scatter(balldist[::skip], playerspeed[::skip])
plt.plot(np.arange(60),speed_reg.coef_[0][0]*np.arange(60)+speed_reg.intercept_[0],'r')
plt.xlabel('Distance to ball (m)')
plt.ylabel('Player speed (m/s)')
plt.show()
plt.savefig(prefix+'speeddist.png')
f = plt.figure(2)
plt.scatter(balldist[::skip], player_relangle[::skip])
plt.show()
plt.savefig(prefix+'relangle.png')
f = plt.figure(3)
plt.plot(np.sort(player_relangle),np.arange(len(player_relangle))/(1.*len(player_relangle)))
plt.show()
plt.savefig(prefix+'angledistribution.png')
f = plt.figure(4)
plt.hist(np.sort(player_relangle),bins=50,normed=True)
plt.xlabel('Angle difference (rad)')
plt.ylabel('Distribution')
plt.ylim(top=0.45)
plt.show()
plt.savefig(prefix+'anglepdf.png')
f = plt.figure(5)
plt.hist(np.sort(player_balldirangle),bins=50,normed=True)
plt.xlabel('Angle difference (rad)')
plt.ylabel('Distribution')
plt.ylim(top=0.45)
plt.show()
plt.savefig(prefix+'balldirangle.png')
f = plt.figure(6)
plt.scatter(ballanglestore[::skip], player_angle[::skip])
plt.plot(np.arange(-3.,3.,0.1),angle_reg.coef_[0][0]*np.arange(-3.,3.,0.1)+angle_reg.intercept_[0],'r')
plt.show()
plt.savefig(prefix+'playerangle_vs_ballangle.png')
f = plt.figure(7)
plt.hist(np.sort(playerspeed),np.arange(13), normed=True)
plt.xlabel('Player speed (m/s)')
plt.ylabel('Distribution')
plt.ylim(top=0.35)
sortedplayerspeed = np.sort(playerspeed.flatten())
l = len(playerspeed)
plt.show()
plt.savefig(prefix+'speedhistogram.png')
