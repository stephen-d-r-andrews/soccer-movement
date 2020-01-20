import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

rawdata = pickle.load(open('../../train_data.pkl','rb'),encoding='latin1')
numseq = 7500

numgoals = 0
numnogoals = 0
goalstart = 0.
nogoalstart = 0.
goaltime = 0.
nogoaltime = 0.
goalnumpass = 0.
nogoalnumpass = 0.
goaldefbehindball = 0.
nogoaldefbehindball = 0.
goalstartarr = []
nogoalstartarr = []
goaltimearr = []
nogoaltimearr = []
goalnumpassarr = []
nogoalnumpassarr = []
goaldefbehindballarr = []
nogoaldefbehindballarr = []

goalspeedarr = []
goaldefspeedarr = []
goalattspeedarr = []
goalballspeedarr = []
goaldiffspeedarr = []
nogoalspeedarr = []
nogoaldefspeedarr = []
nogoalattspeedarr = []
nogoalballspeedarr = []
nogoaldiffspeedarr = []

speedTS = {}
goalSEQ = []
nogoalSEQ = []

stepsize = 1

for i in range(numseq):
#for i in range(100):
    key = 'sequence_'+str(i+1)
    print(key)
    speedTS[key] = []

    seq = rawdata[key]
    goalflag = False
    xmax = 0.
    numpass = 0
    possplayer = -1
    defbehindball = 0
    
    distdef = 0
    distatt = 0
    distball = 0
    for t in range(np.shape(seq)[0]):
        if t==0:
            startx = seq[t,44]
            for d in range(11):
                if seq[t,2*d]>startx:                    
                    defbehindball += 1
        xmax = max(xmax, seq[t,44])
        offplayerdist = np.zeros(11)
        for p in range(11):
            offplayerdist[p] = np.linalg.norm(seq[t,44:46]-seq[t,(22+2*p):(24+2*p)])
        if np.min(offplayerdist)<1.0 and possplayer != np.argmin(offplayerdist):
            possplayer = np.argmin(offplayerdist)
            numpass += 1
        if seq[t,44]>52.5 and seq[t,45]>-3.66 and seq[t,45]<3.66:
            goalflag = True

        if t==0: continue
        delta_coord = (seq[t] - seq[t-1])**2
        delta_dist = [math.sqrt(delta_coord[j]+delta_coord[j+1]) for j in range(0,46,2)]
        distdef += sum(delta_dist[0:11])
        distatt += sum(delta_dist[11:22])
        distball += sum(delta_dist[22:23])

#        distdef += sum(abs(seq[t,0:22] - seq[t-1,0:22]))
#        distatt += sum(abs(seq[t,23:44] - seq[t-1,23:44]))
#        distball += sum(abs(seq[t,44:46] - seq[t-1,44:46]))

#    if xmax>52.5:
        if t < np.shape(seq)[0]-2:
            speedTS[key].append(sum(delta_dist))
            if sum(delta_dist)>30:
                print('huge', key, sum(delta_dist), len(speedTS[key]), len(seq))

    if goalflag:
        numgoals += 1
        goalstart += startx
        goalstartarr.append(startx)
        goaltime += t/10. # freq is 10Hz
        goaltimearr.append(t/10.)
        goalnumpass += numpass
        goalnumpassarr.append(numpass)
        goaldefbehindball += defbehindball
        goaldefbehindballarr.append(defbehindball)

        goalspeedarr.append((distdef+distatt+distball)*(1./(np.shape(seq)[0]-2))*(10./(23.*stepsize)))
        goaldefspeedarr.append(distdef*(1./(np.shape(seq)[0]-2))*(10./(11.*stepsize)))
        goalattspeedarr.append(distatt*(1./(np.shape(seq)[0]-2))*(10./(11.*stepsize)))
        goalballspeedarr.append(distball*(1./(np.shape(seq)[0]-2))*(10./(1.*stepsize)))
        goaldiffspeedarr.append((distdef-distatt)*(1./(np.shape(seq)[0]-2))*(10./(11.*stepsize)))
        goalSEQ.append(i) 

    else:
        numnogoals += 1
        nogoalstart += startx
        nogoalstartarr.append(startx)
        nogoaltime += t/10.
        nogoaltimearr.append(t/10.)
        nogoalnumpass += numpass
        nogoalnumpassarr.append(numpass)
        nogoaldefbehindball += defbehindball
        nogoaldefbehindballarr.append(defbehindball)

        nogoalspeedarr.append((distdef+distatt+distball)*(1./(np.shape(seq)[0]-2))*(10./(23.*stepsize)))
        nogoaldefspeedarr.append(distdef*(1./(np.shape(seq)[0]-2))*(10./(11.*stepsize)))
        nogoalattspeedarr.append(distatt*(1./(np.shape(seq)[0]-2))*(10./(11.*stepsize)))
        nogoalballspeedarr.append(distball*(1./(np.shape(seq)[0]-2))*(10./(1.*stepsize)))
        nogoaldiffspeedarr.append((distdef-distatt)*(1./(np.shape(seq)[0]-2))*(10./(11.*stepsize)))
        nogoalSEQ.append(i) 

print(goalballspeedarr)
if numgoals>0:
    print('goals',numgoals,goalstart/numgoals,goaltime/numgoals,goalnumpass/numgoals,goaldefbehindball/numgoals)
    print('    speeds', sum(goalspeedarr)*10/23/numgoals, sum(goaldefspeedarr)*10/11/numgoals, sum(goalattspeedarr)*10/11/numgoals, sum(goalballspeedarr)*10/numgoals, sum(goaldiffspeedarr)/numgoals)
if numnogoals>0:
    print('nogoals',numnogoals,nogoalstart/numnogoals,nogoaltime/numnogoals,nogoalnumpass/numnogoals,nogoaldefbehindball/numnogoals)
    print('    speeds', sum(nogoalspeedarr)*10/23/numnogoals, sum(nogoaldefspeedarr)*10/11/numnogoals, sum(nogoalattspeedarr)*10/11/numnogoals, sum(nogoalballspeedarr)*10/numnogoals, sum(nogoaldiffspeedarr)/numnogoals)

with open('speedTS.pkl', 'wb') as handle:
    pickle.dump((speedTS), handle, protocol=pickle.HIGHEST_PROTOCOL)

f = plt.figure(0)
x1 = np.sort(np.array(goalstartarr))
y1 = (1./numgoals)*np.arange(numgoals)
print(len(x1), len(y1))
x2 = np.sort(np.array(nogoalstartarr))
y2 = (1./numnogoals)*np.arange(numnogoals)
p1, = plt.plot(x1,y1)
p2, = plt.plot(x2,y2)
plt.xlabel('Ball x-coord at start of possesion (m)')
plt.ylabel('cdf')
plt.legend([p1, p2], ['goal','no goal'],loc='lower right')
plt.show()
plt.savefig('startx.png')

f = plt.figure(1)
x1 = np.sort(np.array(goaltimearr))
y1 = (1./numgoals)*np.arange(numgoals)
print(len(x1), len(y1))
x2 = np.sort(np.array(nogoaltimearr))
y2 = (1./numnogoals)*np.arange(numnogoals)
p1, = plt.plot(x1,y1)
p2, = plt.plot(x2,y2)
plt.xlabel('Time of possesion (s)')
plt.ylabel('cdf')
plt.legend([p1, p2],['goal','no goal'],loc='lower right')
plt.xlim(right=100.)
plt.show()
plt.savefig('time.png')

f = plt.figure(2)
x1 = np.sort(np.array(goalnumpassarr))
y1 = (1./numgoals)*np.arange(numgoals)
print(len(x1), len(y1))
x2 = np.sort(np.array(nogoalnumpassarr))
y2 = (1./numnogoals)*np.arange(numnogoals)
p1, = plt.plot(x1,y1)
p2, = plt.plot(x2,y2)
plt.xlabel('Number of passes in possesion')
plt.ylabel('cdf')
plt.legend([p1, p2], ['goal','no goal'],loc='lower right')
plt.xlim(right=40.)
plt.show()
plt.savefig('numpass.png')

f = plt.figure(3)
x1 = np.sort(np.array(goaldefbehindballarr))
y1 = (1./numgoals)*np.arange(numgoals)
print(len(x1), len(y1))
x2 = np.sort(np.array(nogoaldefbehindballarr))
y2 = (1./numnogoals)*np.arange(numnogoals)
p1, = plt.plot(x1,y1)
p2, = plt.plot(x2,y2)
plt.xlabel('Number of defenders behind ball at start of possesion')
plt.ylabel('cdf')
plt.legend([p1, p2], ['goal','no goal'],loc='lower right')
plt.show()
plt.savefig('numdefbehindball.png')

f = plt.figure(4)
x1 = np.sort(np.array(goalspeedarr))
y1 = (1./numgoals)*np.arange(numgoals)
x2 = np.sort(np.array(nogoalspeedarr))
y2 = (1./numnogoals)*np.arange(numnogoals)
p1, = plt.plot(x1,y1)
p2, = plt.plot(x2,y2)
plt.xlabel('Average speed (m/s)')
plt.ylabel('cdf')
plt.legend([p1, p2], ['goal','no goal'],loc='lower right')
plt.show()
plt.savefig('speed.png')

