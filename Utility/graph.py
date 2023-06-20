import numpy as np
import matplotlib as mpl
mpl.use('pgf')
pgf_with_custom_preamble = {
"text.usetex": True,    # use inline math for ticks
"pgf.rcfonts": False
}
mpl.rcParams.update(pgf_with_custom_preamble)
import matplotlib.pyplot as plt
#plt.style.use(['science','ieee'])
import matplotlib.colors as mcolors
from scipy import stats

def cumulative_pareto_regrets(config,obs_best_arms,save_path):
    plt.clf()
    colors =  list(mcolors.TABLEAU_COLORS.items())
    del colors[3] 
    colors[4], colors[1] = colors[1], colors[4]
    
    marker = [' ', ' ','x', '^', '+', '*', "v", '1', "_"]
    T = range(0,config.T)

    
    pareto_regrets = np.zeros([len(obs_best_arms), config.T])
    obs_best_arms = np.asarray(obs_best_arms)
    for t in T:
        arms_mean = np.asarray(config.environment.get_arms_mean(t))
        ParetoFront = np.ones(config.environment.noArms)
        for j in range(config.environment.noArms):
            for i in range(config.environment.noArms):
                if(np.all(arms_mean[j] >= arms_mean[i]) and np.any(arms_mean[j] > arms_mean[i])):
                    ParetoFront[i] = 0
        #print(ParetoFront)
        for x, obs in enumerate (obs_best_arms):
            Pareto_regrets = 0
            for r in range(config.Rounds):
                mean = arms_mean[int(obs[r,t])]  if config.type == "synthetic" else arms_mean[int(obs[t])]
                deltas = [  np.min(arms_mean[i] - mean) for i in range(config.environment.noArms) if ParetoFront[i] != 0 ]
                deltas = np.max(deltas)
                Pareto_regrets += ( deltas/  config.Rounds )         
            pareto_regrets[x][t] = Pareto_regrets            
                    
    for i, obs in enumerate(obs_best_arms):
        plt.plot(T,(np.cumsum(pareto_regrets[i])) , label=config.algorthims[i].info_str(latex=True), c =colors[i][1],marker =marker[i],markersize=2,markevery=config.T//20 )
        
    print_legend = True
    if(config.environment.Nonstationary ):
        for point in config.environment.arms_breakpoints:
            if print_legend:
                plt.axvline(point, color='red',linestyle='dashed',linewidth = 0.3, label='Break point')
                print_legend = False
            else:
                plt.axvline(point, color='red',linestyle='dashed',linewidth = 0.3)
                
    if(config.type == "synthetic" ) :
            plt.xticks( np.arange(0, config.T + 1, step= (config.T // (len(config.environment.arms_breakpoints )+ 1)) ) , fontsize=8)
            plt.yticks(fontsize=8)
    else:
        plt.ylim([0, 600])
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlim(0,config.T) 
        
    plt.ylabel("Cumulative Pareto Regrets")
    plt.xlabel("Time (t)")
    plt.legend(loc="upper left", fontsize=5, fancybox=True, shadow=False, frameon=True, framealpha=0.8)
    if(save_path == None):
        plt.show()
    else:
        plt.savefig(save_path+ "Cumulative_Pareto_Regrets" +".pdf", dpi = 600)
        plt.savefig(save_path+ "Cumulative_Pareto_Regrets" +".pgf", dpi = 600)
        plt.savefig(save_path+ "Cumulative_Pareto_Regrets" +".png", dpi = 600)


def moving_average(x, w = 60):
    return np.convolve(x, np.ones(w), 'same') / w

def plot_unfairness(config,obs_best_arms,save_path):
    plt.clf()
    
    T = range(0,config.T)
    unfairness = np.zeros([len(obs_best_arms), config.T])
    frontarms = np.zeros([config.environment.noArms, config.T])
    for t in T:
        arms_mean = config.environment.get_arms_mean(t)
        ParetoFront = np.ones(config.environment.noArms)
        for j in range(config.environment.noArms):
            for i in range(config.environment.noArms):
                if(np.all(arms_mean[j] >= arms_mean[i]) and np.any(arms_mean[j] > arms_mean[i])):
                    ParetoFront[i] = 0
                    
        frontarms[:,t] = ParetoFront
                 
                 
        for i, obs in enumerate(obs_best_arms):
            obs = np.asarray(obs)
            for z in range(obs.shape[0]):
                for j in range(config.environment.noArms):
                    unfairness[i,t] +=  np.sum(obs[z,t] == j ) * frontarms[j,t]
            unfairness[i,t] /= (obs.shape[0] * np.sum(frontarms[:,t]) )
                
            
    for i, obs in enumerate(obs_best_arms):        
        plt.plot(T, (np.cumsum(unfairness[i])-T)**2 , label=config.algorthims[i].info_str())
        
    print_legend = True
    if(config.environment.Nonstationary ):
        for point in config.arms["points"]:
            if(point["t1"]>0 and point["t1"] < config.T):
                if print_legend:
                    plt.axvline(point["t1"], color='red',linestyle='dashed', label='Break point')
                    print_legend = False
                else:
                    plt.axvline(point["t1"], color='red',linestyle='dashed')
                
        
    plt.ylabel("unfairness")
    plt.legend(loc="upper left", fontsize=6)
    if(save_path == None):
        plt.show()
    else:
        plt.savefig(save_path+ "unfairness.pdf", dpi = 600)
 
def plot_arms_means(config,save_path):
    plt.clf()
    colors =  list(mcolors.TABLEAU_COLORS.items())

    T = range(0,config.T)
    arms_mean = np.zeros((config.T,config.environment.noArms,config.environment.noObjs))
    for t in T:
        for i, arm in enumerate(config.environment.get_arms_mean(t)):
            arms_mean[t,i] = arm

    fig, axs = plt.subplots(config.environment.noObjs, sharex=True, squeeze=True)
       
    for j in range(config.environment.noObjs):
        for i in range(config.environment.noArms):
            label = "arm " + str(i+1) if i ==0 else str(i+1)
            axs[j].plot(T,arms_mean[:,i,j], c =colors[i][1],label= label)
            axs[j].set_ylabel( 'Obj. '+str(j+1), fontsize=7)
            plt.subplots_adjust(hspace = 0.05 * config.environment.noObjs )
        if(config.type == "synthetic" ) :
            axs[j].set_xticks( np.arange(0, config.T + 1, step= (config.T // (len(config.environment.arms_breakpoints )+ 1)) ))
        axs[j].set_xlim(0,config.T)
    #fig.suptitle("Arms mean")
    lines_labels = [axs[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.subplots_adjust(bottom=0.2)
    plt.figlegend( lines, labels, loc = 'upper center', ncol=5, labelspacing=0.0 )
    plt.xlabel("Time (t)")
    if(config.type != "synthetic" ) :
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if(save_path == None):
        plt.show()
    else:
        plt.savefig(save_path+ "arms_mean.pdf", dpi = 600)
        plt.savefig(save_path+ "arms_mean.pgf", dpi = 600)
        plt.savefig(save_path+ "arms_mean.png", dpi = 600)