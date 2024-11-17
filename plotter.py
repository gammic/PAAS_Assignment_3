import numpy as np
import matplotlib.pyplot as plt


x=[]
y=[]
xgt=[]
ygt=[]
time=[]
fig, (ax1, ax2, ax3) = plt.subplots(3)
#myfile<< best_particle.x<< " "<< best_particle.y<< " " <<gt_x << " "<<gt_y<<" "<<RMSE(0)<<" "<< RMSE(1)<<" " <<duration.count()<<'\n';
with open('res.txt','r') as file:
    # reading each line    
    for line in file:
        # reading each word        
        word = line.split()
        x.append(float(word[0]))
        y.append(float(word[1]))  
        time.append(float(word[2]))

with open('pf_slam.txt','r') as file:
    # reading each line    
    for line in file:
        # reading each word        
        word = line.split()
        xgt.append(float(word[1]))
        ygt.append(float(word[2])) 


ax1.scatter(x, y,color='blue', s=5)
ax2.scatter(xgt, ygt,color='green', s=5)
t=[i for i in range(len(time))]
ax3.plot(t, time)

plt.show()
