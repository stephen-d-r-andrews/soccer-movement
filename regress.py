import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

dataset = 'chs' # 'stats' or 'chs'
if dataset=='stats':
    rawdata=pickle.load(open('train_data.pkl','rb'),encoding='latin1')
    stepsize=1
    seq2process=7500
else:
    rawdata=pickle.load(open('chs_data.pkl','rb'),encoding='latin1')
    stepsize=5
    seq2process=8
vregress_in=[]
vregress_out=[]

print(dataset, seq2process)
#for i in range(seq2process):
for i in range(seq2process):
    key='sequence_'+str(i+1)
#    print(key)
    seq=rawdata[key]
    for t in range(np.shape(seq)[0]-stepsize):
        def legit(t,x):
            if dataset=='stats':
                return True
            if min(seq[t,2*x],seq[t,2*x+1])<-40.:
                return False
            return True

        vball=(seq[t+stepsize,44]-seq[t,44],seq[t+stepsize,45]-seq[t,45])
        for p in range(22):
            vplayer=(seq[t+stepsize,2*p]-seq[t,2*p],seq[t+stepsize,2*p+1]-seq[t,2*p+1])
            dplayerball=(seq[t,44]-seq[t,2*p],seq[t,45]-seq[t,2*p+1])

            closestdefender=-1
            closestattacker=-1
            mindistdefender=np.inf
            mindistattacker=np.inf
            for pd in range(11):
                playerdist=np.linalg.norm((seq[t,2*p]-seq[t,2*pd],seq[t,2*p+1]-seq[t,2*pd+1]))
                if playerdist<mindistdefender and p!=pd and legit(t,pd):
                    mindistdefender=playerdist
                    closestdefender=pd
                    
            for pa in range(11,22):
                playerdist=np.linalg.norm((seq[t,2*p]-seq[t,2*pa],seq[t,2*p+1]-seq[t,2*pa+1]))
                if playerdist<mindistattacker and p!=pa and legit(t,pa):
                    mindistattacker=playerdist
                    closestattacker=pa
            dclosestdefender=(seq[t,2*closestdefender]-seq[t,2*p],seq[t,2*closestdefender+1]-seq[t,2*p+1])
            dclosestattacker=(seq[t,2*closestattacker]-seq[t,2*p],seq[t,2*closestattacker+1]-seq[t,2*p+1])

            def n(v):
                return np.linalg.norm(v)
            if legit(t,22) and legit(t+stepsize,22) and legit(t,p) and legit(t+stepsize,p) and closestdefender>=0 and closestattacker>=0:
                if min([n(vball),n(dplayerball),n(dclosestdefender),n(dclosestattacker),n(vplayer)])>0.:
                    vball=vball/n(vball)
                    dplayerball=dplayerball/n(dplayerball)
                    dclosestdefender=dclosestdefender/n(dclosestdefender)
                    dclosestattacker=dclosestattacker/n(dclosestattacker)
                    vplayer=vplayer/n(vplayer)
                    
                    vregress_in.append([vball[0],vball[1],dplayerball[0],dplayerball[1],dclosestdefender[0],dclosestdefender[1],dclosestattacker[0],dclosestattacker[1]])
#                    vregress_in.append([vball[0],vball[1],dplayerball[0],dplayerball[1]])
                    vregress_out.append([vplayer[0],vplayer[1]])

vregress=LinearRegression().fit(vregress_in,vregress_out)
print('coef',vregress.coef_)
print('intercept',vregress.intercept_)

