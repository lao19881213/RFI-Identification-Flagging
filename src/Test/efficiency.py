#!/usr/bin/env python
import matplotlib.pyplot as plt

fig = plt.figure()
x = [0,1,2,3,4,5,6,7,8,9,10]
y = [0,0.28,0.59,0.98,1.17,1.21,1.25,1.29,1.24,1.30,1.31]
z = [0,0.109,0.203,0.252,0.233,0.240,0.262,0.263,0.261,0.262,0.254]
m = [0,0.258,0.485,0.734,0.889,1.010,1.157,1.333,1.419,1.535,1.590]
n = [0,2.667,4.000,4.752,5.333,4.545,4.800,5.385,4.961,5.143,4.624]
plt.ylim(0,7.5)
plt.plot(x,y,'ko-',label='TH',linewidth=2)
plt.plot(x,z,'ro-',label='GF',linewidth=2)
plt.plot(x,m,'bo-',label='SK',linewidth=2)
plt.plot(x,n,'yo-',label='SVD',linewidth=2)
plt.grid(True)
plt.xlabel('Process(Number)')
plt.ylabel('Speed(Mbit/s)')
plt.legend()
plt.show()
