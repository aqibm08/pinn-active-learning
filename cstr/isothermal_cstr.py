import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy import io
from sklearn.model_selection import train_test_split
np.random.seed(1234)
# make all the plot settings here
plt.rcParams['figure.dpi'] = 180 # this makes higher resolution figures than the default (78)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Arial'
#plt.rcParams["font.weight"] = "bold"
#plt.rcParams["axes.labelweight"] = "bold"
#plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = "medium"
plt.rcParams["axes.titlesize"] = "medium"
plt.rcParams['ytick.labelsize'] = 'medium'
plt.rcParams['xtick.labelsize'] = 'medium'
plt.rcParams['legend.fontsize'] = 14

def isothermal_cstr(t, y,Qf,Qc):
    EbyR=8750    # (K)
    k=7.2e10     # (min^-1)
    Cv = 40.0
    Ca= y[0]
    T = y[1]
    Tc = y[2]
    h = y[3]
    Q = Cv*np.sqrt(h)
    dCa_dt = (-k*np.exp(-EbyR/T)*Ca)+(((Qf*Caf)-(Q*Ca))/(A*h))
    dT_dt = (((k*np.exp(-EbyR/T)*Ca)*(-H))/Rho_Cp)+((Qf*Tf-Q*T)/(A*h))+(U_Ac*(Tc-T)/(Rho_Cp*A*h))
    dTc_dt = ((Qc/Vc)*(Tcf-Tc))+(U_Ac*(T-Tc)/(Rhoc_Cpc*Vc))
    dh_dt = (Qf-Q)/A

    return [dCa_dt,dT_dt,dTc_dt,dh_dt]

def generate_random_signal(nstep,a_range,b_range):
    a = np.random.rand(nstep) * (a_range[1]-a_range[0]) + a_range[0] # range for amplitude
    b = np.random.rand(nstep) *(b_range[1]-b_range[0]) + b_range[0] # range for frequency
    b = np.round(b)
    b = b.astype(int)
    b[0] = 0
    for i in range(1,np.size(b)):
        b[i] = b[i-1]+b[i]

    # Random Signal
    i=0
    random_signal = np.zeros(nstep)
    while b[i]<np.size(random_signal):
        k = b[i]
        random_signal[k:] = a[i]
        i=i+1
    return random_signal

if __name__=='__main__':
    # Define the parameters
    Qc=15        # (L/min)
    Qf=120       # (L/min)
    Tf=320       # (K)
    Q=Qf         # (L/min)
    Tcf=300      # (K)
    T=402.35     # (K)
    Tc=345.44    # (K)
    Caf=1        # (mol/L)
    Ca=0.037     # (mol/L)
    h=0.6     # (m)
    A=100 # (m^2)
    H=-5e4       # (J/mol)
    Rho_Cp=239   # (J/L.K)
    Rhoc_Cpc=4175# (J/L.K)
    U_Ac=5e4     # (J/min.K)
    Vc=250        # (L)
    
    # Initial conditions
    Ca0 = 0.00001
    T0 = 298.15
    Tc0 = 298.15
    h0 = 0.00001
    
    #Generate random signal for Q and Qc
    n_data = 10000
    Q_range = [100,140] #Amplitude range vary between 80 - 120
    Q_freq = [200,250]  # Frequency range
    Qc_range = [10,20] #Amplitude range vary between 10 - 20
    Qc_freq = [200,250] # Frequency range
    Q_random_signal = generate_random_signal(n_data,Q_range,Q_freq)
    Qc_random_signal = generate_random_signal(n_data,Qc_range,Qc_freq)
    
    t = np.arange(0,2000,1)
    t_plot1 = np.arange(0,2000,1)
    t_plot2 = np.arange(2000,5001,1)
    Qf_plot_train = Q_random_signal[:2000]
    Qf_plot_valid = Q_random_signal[-3000:]
    Qf_plot_valid = np.concatenate((np.array([Qf_plot_train[-1]]),Qf_plot_valid))
    Qc_plot_train = Qc_random_signal[:2000]
    Qc_plot_valid = Qc_random_signal[-3000:]
    Qc_plot_valid = np.concatenate((np.array([Qc_plot_train[-1]]),Qc_plot_valid))
    plt.figure()
    #plt.title("Random input signal for data generation")
    plt.plot(t_plot1,Qf_plot_train,'-k',label = '$Q_f$')
    plt.plot(t_plot2,Qf_plot_valid,'--k')
    plt.plot(t_plot1,Qc_plot_train,'-r',label = '$Q_c$')
    plt.plot(t_plot2,Qc_plot_valid,'--r')
    plt.legend()
    plt.grid(linewidth = 0.5)
    plt.xlabel('Time $(min)$')
    plt.xlim = ([0,5000])
    plt.ylabel('Input flowrates $(L/min)$')
    # Time span
    t_span = (0, 1) #For sampling every second
    # Solve the differential equations
    y0 = np.array([Ca0,T0,Tc0,h0])
    sol = solve_ivp(isothermal_cstr, (0,300), y0, method='RK45', t_eval=np.linspace(0, 300, 300),args = (Qf,Qc))
    
    '''#Plotting steady state step response
    fig,axs = plt.subplots(2, 2)
    axs[0,0].plot(sol.t,sol.y[0],'k')
    #axs[0,0].set_xlabel('Time')
    axs[0,0].set_ylabel('$C_A$ $(mol/L)$')
    
    axs[0,1].plot(sol.t,sol.y[1],'k')
    #axs[0,1].set_xlabel('Time')
    axs[0,1].set_ylabel('$T$ $(K)$')
    
    axs[1,0].plot(sol.t,sol.y[2],'k')
    axs[1,0].set_xlabel('Time (min)')
    axs[1,0].set_ylabel('$T_c$ $(K)$')

    axs[1,1].plot(sol.t,sol.y[3],'k')
    axs[1,1].set_xlabel('Time (min)')
    axs[1,1].set_ylabel('$h$ $(m)$')
    plt.subplots_adjust(hspace = 0.3,wspace= 0.3)
    plt.show()'''
    
    # Extract solution
    t = sol.t
    Ca= sol.y[0]
    Ca_ss = Ca[-1]
    T= sol.y[1]
    T_ss = T[-1]
    Tc= sol.y[2]
    Tc_ss = Tc[-1]
    h= sol.y[3]
    h_ss = h[-1]
    
    '''for i in range(len(t)):
        if sol.y[0][i]>0.9995*Ca_ss:
            print(i)
    print('\n')
    for i in range(len(t)):
        if i>50 and sol.y[1][i]<1.0005*T_ss:
            print(i)
    print('\n')
    for i in range(len(t)):
        if i>50 and sol.y[2][i]<1.0005*Tc_ss:
            print(i)
    print('\n')
    for i in range(len(t)):
        if sol.y[3][i]>0.9995*h_ss:
            print(i)
    
    print("Steady state Ca: "+str(Ca_ss))
    print("Steady state T: "+str(T_ss))
    print("Steady state Tc: "+str(Tc_ss))
    print("Steady state h: "+str(h_ss))'''
    
    Ca_vec = np.zeros(len(Q_random_signal))
    T_vec = np.zeros(len(Q_random_signal))
    Tc_vec = np.zeros(len(Q_random_signal))
    h_vec = np.zeros(len(Q_random_signal))
    Ca_vec_init = np.zeros(len(Q_random_signal))
    T_vec_init = np.zeros(len(Q_random_signal))
    Tc_vec_init = np.zeros(len(Q_random_signal))
    h_vec_init = np.zeros(len(Q_random_signal))
    for i in range(len(Q_random_signal)):
        print('i = '+str(i))
        Qf = Q_random_signal[i]
        Qc = Qc_random_signal[i]
        Ca_vec_init[i]=Ca_ss
        T_vec_init[i]=T_ss
        Tc_vec_init[i]=Tc_ss
        h_vec_init[i]=h_ss
        sol = solve_ivp(isothermal_cstr, t_span, [Ca_ss,T_ss,Tc_ss,h_ss], method='RK45', t_eval=np.linspace(0, 1, 100),args = (Qf,Qc))
        t = sol.t
        Ca= sol.y[0]
        Ca_ss = Ca[-1]
        T= sol.y[1]
        T_ss = T[-1]
        Tc= sol.y[2]
        Tc_ss = Tc[-1]
        h= sol.y[3]
        h_ss = h[-1]
        Ca_vec[i]=Ca_ss
        T_vec[i]=T_ss
        Tc_vec[i]=Tc_ss
        h_vec[i]=h_ss
    
    t_vec = np.arange(0,n_data,1)

    x = np.concatenate((Q_random_signal.reshape(n_data,1),Qc_random_signal.reshape(n_data,1)),axis=1)
    x_rnn = np.concatenate((Q_random_signal.reshape(n_data,1),Qc_random_signal.reshape(n_data,1)),axis=1)
    x_encdec = np.concatenate((Q_random_signal.reshape(n_data,1),Qc_random_signal.reshape(n_data,1),Ca_vec_init.reshape(n_data,1),T_vec_init.reshape(n_data,1),Tc_vec_init.reshape(n_data,1),h_vec_init.reshape(n_data,1)),axis=1)
    print(x_encdec.shape)
    y = np.concatenate((Ca_vec.reshape(n_data,1),T_vec.reshape(n_data,1),Tc_vec.reshape(n_data,1),h_vec.reshape(n_data,1)),axis=1)
    
    Q_max = 140
    Qc_max = 20
    Q_min = 100
    Qc_min = 10
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=False)
    savedic = {"x_train":x_train,"x_test":x_test,"y_train":y_train,"y_test":y_test,'Q_max':Q_max,'Qc_max':Qc_max,'Q_min':Q_min,'Qc_min':Qc_min,'tmax':n_data}
    io.savemat("SS_data_ndata10000_sampT1_isothermalcstr.mat",savedic)
    
    #savedic_rnn = {'x':x_rnn,'y':y,'Q_max':Q_max,'Qc_max':Qc_max,'Q_min':Q_min,'Qc_min':Qc_min,'tmax':n_data}
    #io.savemat("SS_data_ndata1000_sampT1_cstr_ForRNN.mat",savedic_rnn)
    
    #savedic_rnn = {'x':x_encdec,'y':y,'Q_max':Q_max,'Qc_max':Qc_max,'Q_min':Q_min,'Qc_min':Qc_min,'tmax':n_data}
    #io.savemat("SS_data_ndata1000_sampT1_cstr_ForEncDec.mat",savedic_rnn)
    
    Ca_plot_train = Ca_vec[:2000]
    Ca_plot_valid = Ca_vec[-3000:]
    Ca_plot_valid = np.concatenate((np.array([Ca_plot_train[-1]]),Ca_plot_valid))
    T_plot_train = T_vec[:2000]
    T_plot_valid = T_vec[-3000:]
    T_plot_valid = np.concatenate((np.array([T_plot_train[-1]]),T_plot_valid))
    Tc_plot_train = Tc_vec[:2000]
    Tc_plot_valid = Tc_vec[-3000:]
    Tc_plot_valid = np.concatenate((np.array([Tc_plot_train[-1]]),Tc_plot_valid))
    h_plot_train = h_vec[:2000]
    h_plot_valid = h_vec[-3000:]
    h_plot_valid = np.concatenate((np.array([h_plot_train[-1]]),h_plot_valid))
    #Plotting
    fig,axs = plt.subplots(2, 2)
    #plt.suptitle('Simulation data')
    axs[0,0].plot(t_plot1,Ca_plot_train,'-k',label = 'Training')
    axs[0,0].plot(t_plot2,Ca_plot_valid,'--k',label = 'Validation')
    #axs[0,0].set_xlabel('Time')
    axs[0,0].legend(fontsize = 9)
    axs[0,0].grid(linewidth = 0.5)
    axs[0,0].set_ylabel('$C_A$ $(mol/L)$')
    
    axs[0,1].plot(t_plot1,T_plot_train,'-k',label = 'Training')
    axs[0,1].plot(t_plot2,T_plot_valid,'--k',label = 'Validation')
    axs[0,1].grid(linewidth = 0.5)
    #axs[0,1].set_xlabel('Time')
    axs[0,1].set_ylabel('$T$ $(K)$')
    
    axs[1,0].plot(t_plot1,Tc_plot_train,'-k',label = 'Training')
    axs[1,0].plot(t_plot2,Tc_plot_valid,'--k',label = 'Validation')
    axs[1,0].grid(linewidth = 0.5)
    axs[1,0].set_xlabel('Time (min)')
    axs[1,0].set_ylabel('$T_c$ $(K)$')

    axs[1,1].plot(t_plot1,h_plot_train,'-k',label = 'Training')
    axs[1,1].plot(t_plot2,h_plot_valid,'--k',label = 'Validation')
    axs[1,1].grid(linewidth = 0.5)
    axs[1,1].set_xlabel('Time (min)')
    axs[1,1].set_ylabel('$h$ $(m)$')
    plt.subplots_adjust(hspace = 0.3,wspace= 0.3)
    
    plt.show()
    
    
