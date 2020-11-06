import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from pylab import genfromtxt; 
import matplotlib.ticker as ticker
from scipy import stats 
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 3.1
mpl.rcParams['axes.linewidth'] = 3.1
plt.rcParams['font.family'] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.bottom'] = True
mpl.rcParams['ytick.left'] = True
def as_si(x, ndp):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))
from matplotlib.ticker import ScalarFormatter

data1 = np.loadtxt("data_Flu_Single.txt")
Inf_flu = data1[:,3] 
t_flu = data1[:,0] 

data2 = np.loadtxt("data_SARS_Single.txt")
Inf_sars = data2[:,3] 
t_sars = data2[:,0] 

#parameters
N=331002651
####IAV
#b1, k1, d1=0.001, 0.125, 0.11
#b1, k1, d1=1.0, 0.25, 0.2
b1, k1, d1=0.5/N, 0.25, 0.2 ### Influenza 
b2, k2, d2=0.41/N, 0.2, 0.1 ### SARS-CoV-2
d3=  0.1
# initial conditions
S0, E01, E02, E03, I01, I02, I03, RS01, RS02, RL01, RL02, RI01, RI02, R0= N, 0, 0, 0, 100, 100, 0, 0,0,0,0,0,0,0
# initial condition vector
y0 = [S0, E01, E02, E03, I01, I02, I03, RS01, RS02, RL01, RL02, RI01, RI02, R0]  
#Time : how long to simulate the model
t= np.linspace(0,1000,1000)
   
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
    return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13]
#Solve the model(integrate)------------------------------------
ds = integrate.odeint(funct,y0,t)


data=np.column_stack((t,ds[:,0],ds[:,1],ds[:,2],ds[:,3],ds[:,4],ds[:,5],ds[:,6],ds[:,7],ds[:,8],ds[:,9],ds[:,10],ds[:,11],ds[:,12],ds[:,13]))
np.savetxt("data_coinfection_dynamics_orig.dat",data) 
data=np.column_stack((t,ds[:,4],ds[:,5],ds[:,6],ds[:,11],ds[:,12],ds[:,4]+ds[:,12],ds[:,5]+ds[:,11]))
np.savetxt("infected_orig.dat",data)
data=np.column_stack((t,(ds[:,4]+ds[:,12])*0.1,(ds[:,5]+ds[:,11])*0.3,((ds[:,4]+ds[:,12])*0.1+(ds[:,5]+ds[:,11]+ds[:,6])*0.3)))
np.savetxt("hospital_orig.dat",data)
#print(ds[-1,14]/N,ds[-1,15]/N,ds[-1,16]/N)

##Plot 
fig=plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
plt.plot(t, np.log10(ds[:,1]), 'r',t, np.log10(ds[:,2]), 'b',t, np.log10(ds[:,3]), 'c',linewidth=6)
plt.xlim(1,500)
plt.ylim(0,14)
plt.legend(('Exposed to Flu', 'Exposed to CoV-2','First exposed to''\n' 'CoV-2, then to Flu',),loc='best',fontsize=34)
plt.xlabel('Time (days)', fontsize=45)
plt.ylabel(r'Exposed population (log$_{10}$)', fontsize=45)
plt.tick_params(axis='both', which='major', labelsize=45)
plt.tick_params(axis='both', which='minor', labelsize=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(90))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.tick_params(direction='in', length=8, width=2, colors='k',
               grid_color='k')
#plt.savefig("exposedclass_d3_01.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()


fig=plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
plt.plot(t, np.log10(ds[:,4]+ds[:,12]), 'r',t, np.log10(ds[:,5]+ds[:,11]), 'b',t, np.log10(ds[:,6]), 'c',linewidth=6)
plt.plot(t_flu, np.log10(Inf_flu), 'pink',linestyle=':',linewidth=5)
plt.plot(t_sars, np.log10(Inf_sars), 'skyblue',linestyle=':',linewidth=5)
plt.xlim(1,500)
plt.ylim(0,14)
plt.legend(('Flu', 'CoV-2','Coinfected', 'Single infection: Flu', 'Single infection: CoV-2',),loc='best',fontsize=30)
plt.xlabel('Time (days)', fontsize=45)
plt.ylabel(r'Infected population (log$_{10}$)', fontsize=45)
plt.tick_params(axis='both', which='major', labelsize=45)
plt.tick_params(axis='both', which='minor', labelsize=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(90))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.tick_params(direction='in', length=8, width=2, colors='k',
               grid_color='k')
#plt.savefig("infectedclass_d3_01.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()

fig=plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
plt.plot(t, np.log10(ds[:,4]+ds[:,5]+ds[:,6]+ds[:,11]+ds[:,12]), 'darkorange',linestyle='-',linewidth=5)
plt.plot(t_flu, np.log10(Inf_flu+Inf_sars), 'darkorange',linestyle=':',linewidth=5)
plt.xlim(1,500)
plt.ylim(0,14)
plt.legend(('Total infected with coinfection','Total infected without coinfection',),loc='best',fontsize=30)
plt.xlabel('Time (days)', fontsize=45)
plt.ylabel(r'Infected population (log$_{10}$)', fontsize=45)
plt.tick_params(axis='both', which='major', labelsize=45)
plt.tick_params(axis='both', which='minor', labelsize=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(120))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.tick_params(direction='in', length=8, width=2, colors='k',
               grid_color='k')
#plt.savefig("total_infectedclass_d3_01.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()


fig=plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
plt.plot(t, np.log10(ds[:,7]), 'k', t, np.log10(ds[:,8]), 'gray', linewidth=6 )
plt.xlim(1,500)
plt.ylim(0,14)
plt.legend(('Recovered from Flu,''\n' 'susceptible to CoV-2', 'Recovered from CoV-2,''\n' 'susceptible to Flu',),loc='best',fontsize=34)
plt.xlabel('Time (days)', fontsize=45)
plt.ylabel(r'Recovered susceptible (log$_{10}$)', fontsize=45)
plt.tick_params(axis='both', which='major', labelsize=45)
plt.tick_params(axis='both', which='minor', labelsize=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(90))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.tick_params(direction='in', length=8, width=2, colors='k',
               grid_color='k')
#plt.savefig("recoveredsusceptible_d3_01.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()

fig=plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
plt.plot(t, np.log10(ds[:,9]), 'maroon', t, np.log10(ds[:,10]), 'darkviolet', linestyle='-', linewidth=6)
plt.xlim(1,500)
plt.ylim(0,14)
plt.legend(('Recovered from Flu,''\n' 'exposed to CoV-2', 'Recovered from CoV-2,''\n' 'exposed to Flu',),loc='best',fontsize=34)
plt.xlabel('Time (days)', fontsize=45)
plt.ylabel(r'Recovered exposed (log$_{10}$)', fontsize=45)
plt.tick_params(axis='both', which='major', labelsize=45)
plt.tick_params(axis='both', which='minor', labelsize=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(90))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.tick_params(direction='in', length=8, width=2, colors='k',
               grid_color='k')
#plt.savefig("recoveredexposed_d3_01.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()


fig=plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
plt.plot(t, np.log10(ds[:,11]), 'maroon',t, np.log10(ds[:,12]), 'darkviolet',linewidth=6)
plt.xlim(1,500)
plt.ylim(0,14)
plt.legend(('Recovered from Flu,' '\n''infected with CoV-2', 'Recovered from CoV-2,''\n' 'infected with Flu',),loc='best',fontsize=34)
plt.xlabel('Time (days)', fontsize=45)
plt.ylabel(r'Recovered infected (log$_{10}$)', fontsize=45)
plt.tick_params(axis='both', which='major', labelsize=45)
plt.tick_params(axis='both', which='minor', labelsize=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(90))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.tick_params(direction='in', length=8, width=2, colors='k',
               grid_color='k')
#plt.savefig("recoveredinfected_d3_01.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()

fig=plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
plt.plot(t, np.log10(ds[:,0]), 'g',t, np.log10(ds[:,13]), 'm',linewidth=6)
plt.xlim(1,500)
plt.ylim(0,14)
plt.legend(('Susceptible', 'Recovered',),loc='best',fontsize=34)
plt.xlabel('Time (days)', fontsize=45)
plt.ylabel(r'Population (log$_{10}$)', fontsize=45)
plt.tick_params(axis='both', which='major', labelsize=45)
plt.tick_params(axis='both', which='minor', labelsize=45)
ax.xaxis.set_major_locator(ticker.MultipleLocator(90))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.tick_params(direction='in', length=8, width=2, colors='k',
               grid_color='k')
#plt.savefig("population_d3_01.pdf", bbox_inches='tight', pad_inches=0.2)
plt.show()
