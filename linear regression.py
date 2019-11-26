import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

df_boston = datasets.load_boston()
data = df_boston.data
target = df_boston.target

print("Features:    ",df_boston.feature_names)
print("Data size:   ", data.shape)
print("Target size: ",target.shape)
print("target:      "   ,target)



def estimate_coef(x, y): 
    n = np.size(x)

    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 

    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 

    # print ("regression coefficients", )
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
    
    return(b_0, b_1)




plt.figure()
fig,ax=plt.subplots(1,2,figsize=(21, 10))
n=data.shape[1]
bz, bo = np.zeros(n), np.zeros(n)
for i in range(0,n):
    bz[i], bo[i]= estimate_coef(target,data[:,i])
    print(df_boston.feature_names[i] ,bz[i], bo[i])
    
plt.scatter(data[:,0], target)
z = np.polyfit(data[:,0],target, 1)
p = np.poly1d(z)
plt.plot(data[:,0],p(data[:,0]),"k--", linewidth=2)  

plt.show()
plt.close()


