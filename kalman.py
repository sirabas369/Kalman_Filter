from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import copy

msr = np.genfromtxt('/home/sirabas/Documents/AGV/Task_5/kalmann.txt' , delimiter = ',' , skip_header = 1)
init_msr = np.genfromtxt('/home/sirabas/Documents/AGV/Task_5/kalmann.txt' , delimiter = ',', max_rows = 1 )

data = np.zeros((msr.shape[0]+1 , 8))
data[0 , 0:2] = copy.deepcopy(init_msr)
data[1: , 0:4] = copy.deepcopy(msr)

dt = 1

I = np.array(  [[1,0,0,0], 
                [0,1,0,0], 
                [0,0,1,0], 
                [0,0,0,1]] ) 

X = np.array(  [[0], #posx
                [0], #velx
                [0], #posy
                [0]] ) #vely

P = np.array(  [[5,0,0,0], 
                [0,500,0,0], 
                [0,0,5,0], 
                [0,0,0,500]] ) 

H = np.array(  [[1,0,0,0], 
                [0,1,0,0], 
                [0,0,1,0], 
                [0,0,0,1]] ) 

q = np.array(  [[(dt**4)/4, (dt**3)/2, 0,         0        ], 
                [(dt**3)/2,  dt**2,    0,         0        ], 
                [0,          0,        (dt**4)/4, (dt**3)/2], 
                [0,          0,        (dt**3)/2, dt**2    ]] ) 

Q = q*0.01

F = np.array(  [[1,dt,0,0], 
                [0,1,0,0], 
                [0,0,1,dt], 
                [0,0,0,1]] ) 

R = np.array(  [[1.5,0,0,0], 
                [0,0.1,0,0], 
                [0,0,1.5,0], 
                [0,0,0,0.05]] ) 


for i in range(data.shape[0]):

    M = np.array(  [[data[i,0]],
                    [data[i,2]],
                    [data[i,1]],
                    [data[i,3]]]  )

    if(i == 0):
        X = M
        X = np.dot(F, X)
        P = np.linalg.multi_dot([F, P, F.T]) + Q
        print('At i = ',i)
        print(X)
        print(P)
        data[i,4] = X[0,0]
        data[i,5] = X[2,0]
        data[i,6] = X[1,0]
        data[i,7] = X[3,0]
        continue

    Z = np.dot(H, M) 
    #Z = np.dot(H, M) 
    K = np.linalg.multi_dot([P, H.T ,np.linalg.inv(np.linalg.multi_dot([H, P, H.T]) + R)])
    X = X + np.dot(K , (Z - np.dot(H,X)))
    a = (I - np.dot(K,H))
    P = np.linalg.multi_dot([a, P, a.T])

    data[i,4] = X[0,0]
    data[i,5] = X[2,0]
    data[i,6] = X[1,0]
    data[i,7] = X[3,0]
    print(f'At i = {i}')
    print(K)
    print(X)
    print(P)
    
    X = np.dot(F, X) 
    P = np.linalg.multi_dot([F, P, F.T]) + Q

plt.figure(1)
plt.plot(data[:,0] , color = 'r' , label = 'Measured x values')
plt.plot(data[:,4] , color = 'g' , label = 'Estimated x values')
plt.legend()

plt.figure(2)
plt.plot(data[:,1] , color = 'r' , label = 'Measured y values')
plt.plot(data[:,5] , color = 'g' , label = 'Estimated y values')
plt.legend()

plt.figure(3)
plt.plot(data[:,2] , color = 'r' , label = 'Measured Vx values')
plt.plot(data[:,6] , color = 'g' , label = 'Estimated Vx values')
plt.legend()

plt.figure(4)
plt.plot(data[:,3] , color = 'r' , label = 'Measured Vy values')
plt.plot(data[:,7] , color = 'g' , label = 'Estimated Vy values')
plt.legend()

plt.figure(5)
plt.plot(data[:,4], data[:,5],  color = 'g')
#plt.plot(data[:,0], data[:,1],  color = 'r')


plt.show()