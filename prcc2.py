import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from pylab import genfromtxt; 
import matplotlib.ticker as ticker
#from scipy import stats 
import matplotlib as mpl
import random
from scipy import stats, linalg
from matplotlib.ticker import ScalarFormatter

mpl.rcParams['axes.linewidth'] = 3.1
mpl.rcParams['axes.linewidth'] = 3.1
plt.rcParams['font.family'] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.left'] = True
#parameters
#global N, b1, k1, d1, b2, k2, d2, h_iav, h_covid
N=331002651
####IAV
#b1, k1, d1=0.001, 0.125, 0.11
#b1, k1, d1=1.0, 0.25, 0.2
b1_base, k1_base, d1_base=0.5/N, 0.25, 0.2 ### Influenza 
b2_base, k2_base, d2_base=0.41/N, 0.2, 0.1 ### SARS-CoV-2
d3_base=  0.1
h_iav_base, h_covid_base=0.1, 0.3
# initial conditions
S0, E01, E02, E03, I01, I02, I03, RS01, RS02, RL01, RL02, RI01, RI02, R0, FI0, CI0, FC0 = N, 0, 0, 0, 100, 100, 0, 0,0,0,0,0,0,0,0,0,0
# initial condition vector
y0 = [S0, E01, E02, E03, I01, I02, I03, RS01, RS02, RL01, RL02, RI01, RI02, R0, FI0, CI0, FC0]  
#Time : how long to simulate the model
#t= np.linspace(0,1000,1000)
   
#the model equations 
def funct(y,t):
    S=y[0]
    E1=y[1]
    E2=y[2]
    E3=y[3]
    I1=y[4]
    I2=y[5]
    I3=y[6]
    RS1=y[7]
    RS2=y[8]
    RL1=y[9]
    RL2=y[10]
    RI1=y[11]
    RI2=y[12]
    R=y[13]
    FI=y[14]
    CI=y[15]
    FC=y[16]
    ############
    f0 = - b1*S*(I1+I3+RI2) - b2*S*(I2+I3+E3+RI1)
    #######Exposed Class
    f1 = b1*S*(I1+I3+RI2) - k1*E1
    f2 = b2*S*(I2+I3+E3+RI1) - k2*E2 - b1*E2*(I1+I3+RI2)
    f3 = b1*(E2+I2)*(I1+I3+RI2) -k1*E3 
    ####### Infected Class
    f4 = k1*E1 - d1*I1 ##flu single infected
    f5 = k2*E2 - d2*I2 -b1*I2*(I1*I3+RI2) ##Covid-2 infected
    f6 = k1*E3 - d3*I3  ## coinfected
    ####### Recovered susceptible Class
    f7 = d1*I1 - b2*RS1*(I2+I3+E3+RI1)
    f8 = d2*I2 - b1*RS2*(I1+I3+RI2)
    ############ Recovered Exposed Class
    f9 = b2*RS1*(I2+I3+E3+RI1) - k2*RL1
    f10 =b1*RS2*(I1+I3+RI2) - k1*RL2
    ############ Recovered Infected Class
    f11 =k2*RL1 -d2*RI1 ##Covid infected
    f12 =k1*RL2 -d1*RI2 ## flu infected
    ############ Recovered Class
    f13 =d2*RI1+d1*RI2+d3*I3
    f14 = b1*S*(I1+I3+RI2) + b1*RS2*(I1+I3+RI2)
    f15 = b2*S*(I2+I3+E3+RI1) + b2*RS1*(I2+I3+E3+RI1) - b1*E2*(I1+I3+RI2) - b1*I2*(I1*I3+RI2)
    f16 = b1*E2*(I1+I3+RI2) + b1*I2*(I1*I3+RI2)
    return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16]


def partial_corr(C):
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]
            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr

C=np.zeros([1000,10])

for i in range(1000):
    a=0.1 #10%
    b1 = random.uniform(b1_base-b1_base*a,b1_base+b1_base*a)
    k1 = random.uniform(k1_base-k1_base*a,k1_base+k1_base*a)
    d1 = random.uniform(d1_base-d1_base*a,d1_base+d1_base*a)
    b2 = random.uniform(b2_base-b2_base*a,b2_base+b2_base*a)
    k2 = random.uniform(k2_base-k2_base*a,k2_base+k2_base*a)
    d2 = random.uniform(d2_base-d2_base*a,d2_base+d2_base*a)
    d3 = random.uniform(d3_base-d3_base*a,d3_base+d3_base*a)
#    h_iav= random.uniform(h_iav_base-h_iav_base*a,h_iav_base+h_iav_base*a)
#    h_covid= random.uniform(h_covid_base-h_covid_base*a,h_covid_base+h_covid_base*a)
    
    t= np.linspace(0,1000,1000)
    ds = integrate.odeint(funct,y0,t)
    
#    IAVpeak=np.argmax(ds[:,4]+ds[:,12])
#    CVpeak=np.argmax(ds[:,5]+ds[:,11])
#    Length=CVpeak-IAVpeak
#    print(Length)
#    TimeBetpeaks=[]
#    TimeBetpeaks.append(abs(Length))
#    TimeBetpeaks.append(Length)
#    Timebet=np.array(TimeBetpeaks)
    
#    Infected=h_iav*(ds[:,4]+ds[:,12])+h_covid*(ds[:,5]+ds[:,11]+ds[:,6])
#    Total_hospitalized=np.trapz(Infected,t)
#    Infected=ds[:,4]+ds[:,12]+ds[:,5]+ds[:,11]+ds[:,6]
#    Total_hospitalized=np.trapz(Infected,t)
#    Hospitalized=[]
#    Hospitalized.append(Total_hospitalized)
#    Hos=np.array(Hospitalized)
    
    C[i,:]= [b1,k1,d1,b2,k2,d2,d3,ds[-1,14]/N,ds[-1,15]/N,ds[-1,16]/N]

#
#print(partial_corr(C))
data=np.column_stack((partial_corr(C)))
np.savetxt("data_PC_new1.txt",data) 
