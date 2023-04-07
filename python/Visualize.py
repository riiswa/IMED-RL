import numpy as np
from matplotlib import pyplot as plt
import scipy.special
import matplotlib
from matplotlib import patheffects as pe
plt.rcParams["figure.figsize"] = (12,12)
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.loc'] = 'upper left'




mean_w_10_750_15 = np.array([0.0149, 0.008, 0.007, 0.007, 0.06, 0.35, 0.544, 0.498, 0.57, 0.61, 0.699, 0.7, 0.78, 0.63, 0.82, 0.7, 0.88, 0.86])
ci__10_750_15 = np.array([0.007, 0.002, 0.0008, 0.0004, 0.022, 0.05, 0.08, 0.07, 0.07, 0.07, 0.07, 0.07, 0.101, 0.06, 0.09, 0.1, 0.08, 0.1])

mean_50_750_15 = np.array([ 0.009, 0.006, 0.01, 0.008, 0.039, 0.36, 0.37, 0.60, 0.61, 0.56, 0.63, 0.779, 0.72, 0.71, 0.59, 0.83, 0.93, 0.76])
ci_50_750_15 = np.array([0.002, 0.0002, 0.003, 0.001, 0.01, 0.06, 0.06, 0.07, 0.07,  0.06, 0.06, 0.09, 0.07, 0.07, 0.06, 0.09, 0.1, 0.07])

mean_750_750_15 = np.array([0.01, 0.006, 0.006, 0.02, 0.07, 0.46, 0.42, 0.41, 0.62, 0.74, 0.74, 0.7, 0.69, 0.8, 0.68, 0.766, 0.95, 0.84 ])
ci_750_750_15 = np.array([0.007, 0.0002, 0.0002, 0.015, 0.025, 0.09, 0.05, 0.04, 0.08, 0.11, 0.08, 0.08, 0.008, 0.09, 0.07, 0.08, 0.11, 0.1])

mean_50_15 = np.array([0.01, 0.006, 0.01, 0.01, 0.009, 0.01, 0.04, 0.12, 0.2, 0.21, 0.41, 0.47, 0.52, 0.35, 0.49, 0.48, 0.56, 0.48])
ci_50_15= np.array([0.005, 0.0001, 0.004, 0.007, 0.001, 0.002, 0.03, 0.04, 0.06, 0.04, 0.08, 0.09, 0.06, 0.05, 0.06, 0.07, 0.07, 0.06])

mean_25_15 = np.array([0.02, 0.03, 0.006, 0.013, 0.007, 0.009, 0.03, 0.19, 0.22, 0.18, 0.29, 0.40, 0.38, 0.51, 0.57, 0.53, 0.5, 0.61])
ci_25_15= np.array([0.01, 0.02, 0.0002, 0.004, 0.0001, 0.0009, 0.02, 0.07, 0.05, 0.03, 0.05, 0.03, 0.05, 0.07, 0.06, 0.07, 0.064, 0.07])

mean_10_15 = np.array([0.007, 0.006, 0.01, 0.01, 0.0076, 0.007, 0.01, 0.02, 0.31, 0.39, 0.22, 0.48, 0.44, 0.51, 0.47, 0.61, 0.64, 0.46])
ci_10_15= np.array([0.0003, 0.0001, 0.006, 0.007, 0.00001, 0.0001, 0.004, 0.02, 0.06, 0.06, 0.04, 0.08, 0.08, 0.08, 0.07, 0.08, 0.1, 0.05])

mean_p_40_15 = np.array([0.006, 0.006, 0.018, 0.014, 0.011, 0.06, 0.016, 0.008, 0.03, 0.07, 0.06, 0.17, 0.31, 0.35, 0.51, 0.67, 0.74, 0.76])
ci_p_40_15= np.array([0.0003, 0.0001, 0.011, 0.006, 0.003, 0.0002, 0.01, 0.001, 0.013, 0.037, 0.018, 0.045, 0.067, 0.05, 0.06, 0.085, 0.08, 0.08])

mean_p_56568_15 = np.array([0.007, 0.007, 0.008, 0.006, 0.007, 0.018, 0.006, 0.06, 0.05, 0.09, 0.12, 0.24, 0.32, 0.40, 0.57, 0.64, 0.64, 0.85])
ci_p_56568_15= np.array([0.0003, 0.0007, 0.001, 0.00016, 0.00075, 0.011, 0.0002, 0.02, 0.01, 0.03, 0.03, 0.05, 0.06, 0.06, 0.07, 0.09, 0.07, 0.11])

mean_p_70_15 = np.array([0.009, 0.006, 0.006, 0.008, 0.006, 0.002,0.002, 0.024, 0.04, 0.03, 0.12, 0.244, 0.254, 0.46, 0.42, 0.49, 0.59, 0.68])
ci_p_70_15= np.array([0.002, 0.0002, 0.0003, 0.001, 0.0001, 0.01, 0.01, 0.007, 0.01, 0.01, 0.03, 0.05, 0.044, 0.07, 0.06, 0.05, 0.06, 0.07])

x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
plt.xticks([20000, 40000, 60000, 80000, 100000, 120000])
#plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15.0, 17.5])
plt.xlabel("Number of Simulations per Step")
plt.ylabel("Discounted Return")
plt.title("RiverSwim 95% CI, 50 seeds")

lw = 3
plt.plot(x, mean_50_750_15, "navy", label="W-MCTS-OS -std=5.0 -C=7.5 -p=1.5", linestyle= 'solid', path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_50_750_15 - ci_50_750_15), (mean_50_750_15+ ci_50_750_15), color='navy', alpha = 0.1)

plt.plot(x, mean_750_750_15, "royalblue", label="W-MCTS-OS -std=7.5 -C=7.5 -p=1.5", linestyle= 'dashed', path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_750_750_15- ci_750_750_15), (mean_750_750_15+ ci_750_750_15), color='royalblue', alpha = 0.1)

plt.plot(x, mean_w_10_750_15, "blue", label="W-MCTS-OS -std=10.0 -C=7.5 -p=1.5", linestyle= 'dotted', path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_10_750_15 - ci__10_750_15), (mean_w_10_750_15+ ci__10_750_15), color='blue', alpha = 0.1)

plt.plot(x, mean_25_15, "turquoise", label="W-MCTS-TS -std=2.5 -p=1.5", linestyle= 'solid', path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_25_15 - ci_25_15), (mean_25_15+ ci_25_15), color='turquoise', alpha = 0.1)

plt.plot(x, mean_50_15, "teal", label="W-MCTS-TS -std=5.0 -p=1.5", linestyle= 'dashed', path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_50_15 - ci_50_15), (mean_50_15+ ci_50_15), color='teal', alpha = 0.1)

plt.plot(x, mean_10_15, "mediumaquamarine", label="W-MCTS-TS -std=10.0 -p=1.5", linestyle= 'dotted', path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_10_15 - ci_10_15), (mean_10_15+ ci_10_15), color='mediumaquamarine', alpha = 0.1)

plt.plot(x, mean_p_40_15, "red", label="Power-UCT -C=4.0 -p=1.5", linestyle= 'solid', path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_p_40_15 - ci_p_40_15), (mean_p_40_15+ ci_p_40_15), color='red', alpha = 0.1)

plt.plot(x, mean_p_56568_15, "maroon", label="Power-UCT -C=5.6568 -p=1.5", linestyle= 'dashed', path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_p_56568_15 - ci_p_56568_15), (mean_p_56568_15+ ci_p_56568_15), color='maroon', alpha = 0.1)

plt.plot(x, mean_p_70_15, "indianred", label="Power-UCT -C=7.0 -p=1.5", linestyle= 'dotted', path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_p_70_15 - ci_p_70_15), (mean_p_70_15+ ci_p_70_15), color='indianred', alpha = 0.1)

plt.ylim(bottom=0, top=1.2)
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.savefig('riverswim_res.pdf')
plt.clf()



mean_w_125_300_15 = np.array([1.72, 1.49, 1.61, 1.65, 2.02, 2.5, 2.9, 3.2, 4.0, 4.5, 4.6, 5.0, 5.4, 5.6, 5.5, 5.7, 5.4, 5.6])
ci_125_300_15 = np.array([0.08, 0.07, 0.1, 0.11, 0.17, 0.19, 0.22, 0.27, 0.29, 0.33, 0.31, 0.309, 0.27, 0.3,0.299, 0.3, 0.28, 0.3])

mean_150_300_15 = np.array([1.62, 1.56,1.64, 2.07, 2.27, 2.955, 3.32, 3.81, 3.4, 3.9, 4.7, 4.9, 5.2, 4.5, 5.2, 5.02, 5.29, 5.33])
ci_150_300_15 = np.array([0.07, 0.08, 0.09, 0.17, 0.15, 0.24, 0.25, 0.24, 0.209, 0.26, 0.25, 0.32, 0.29, 0.27, 0.31, 0.26, 0.31, 0.25])

mean_150_50_15 = np.array([1.57, 1.27, 1.799, 1.653, 2.19, 2.47, 2.64, 3.48, 4.08, 4.36, 4.54, 5.05, 5.24, 5.71, 5.25, 5.66, 5.4, 5.6 ])
ci_150_50_15 = np.array([0.1, 0.05, 0.13, 0.11, 0.18, 0.208, 0.21, 0.26, 0.28, 0.308, 0.33, 0.32, 0.28, 0.33, 0.33, 0.33, 0.29, 0.29])

mean_175_15 = np.array([1.4295, 1.44, 1.71, 1.70, 2.33, 3.38, 3.73, 4.59, 4.56, 4.9, 5.16, 5.034, 5.12, 5.17, 5.24, 5.13, 5.22, 5.15])
ci_175_15= np.array([0.08, 0.08, 0.12, 0.13, 0.23, 0.26, 0.24, 0.28, 0.25, 0.27, 0.31, 0.32, 0.31, 0.26, 0.34, 0.26, 0.28, 0.34])

mean_15_15 = np.array([1.66, 1.37, 1.53, 1.87, 2.05, 2.8, 4.19, 3.85, 4.8, 4.8, 5.09, 5.4, 4.99, 5.3, 5.4, 5.34, 4.65, 5.79])
ci_15_15= np.array([0.12, 0.07, 0.14, 0.19, 0.19, 0.17, 0.28, 0.21, 0.29, 0.32, 0.32, 0.32, 0.304, 0.25, 0.33, 0.31, 0.24, 0.36])

mean_1250_15 = np.array([1.59, 1.41, 1.58, 1.55, 2.28, 2.7, 4.03, 4.37, 4.74, 4.58, 5.26, 5.14, 5.53, 5.47, 5.14, 5.55, 5.22, 5.36])
ci_1250_15= np.array([0.08, 0.09, 0.11, 0.07, 0.17, 0.18, 0.28, 0.24, 0.31, 0.29, 0.28, 0.29, 0.32, 0.32, 0.33, 0.34, 0.28, 0.33])

mean_p_200_15 = np.array([1.64, 1.48, 1.51, 1.62, 1.88, 1.61, 2.23, 1.96, 1.88, 1.82, 1.52, 1.71, 1.55, 1.55, 1.58, 1.55, 1.57, 1.59])
ci_p_200_15= np.array([0.09, 0.09, 0.07, 0.09, 0.14, 0.11, 0.19, 0.13, 0.11, 0.11, 0.1, 0.1, 0.08, 0.1, 0.08, 0.08, 0.1, 0.11])

mean_p_50_15 = np.array([1.55, 1.50, 1.56, 1.75, 1.91, 1.64, 2.06, 1.95, 1.86, 1.59, 1.71, 1.33, 1.68, 1.58, 1.51, 1.58, 1.6, 1.4])
ci_p_50_15= np.array([0.08, 0.07, 0.12, 0.11, 0.12, 0.08, 0.12, 0.14, 0.15, 0.11, 0.11, 0.07, 0.1, 0.1, 0.08, 0.08, 0.09, 0.07])

mean_p_300_15 = np.array([1.5918, 1.46, 1.71, 1.95, 1.99, 1.85, 1.81, 2.05, 1.76, 1.63, 1.72, 1.55, 1.67, 1.66, 1.46, 1.84, 1.5, 1.44])
ci_p_300_15= np.array([0.08, 0.07, 0.11, 0.18, 0.13, 0.13, 0.11, 0.17, 0.14, 0.12, 0.15, 0.09, 0.11, 0.13, 0.106, 0.13, 0.09, 0.05])

x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
plt.xticks([20000, 40000, 60000, 80000, 100000, 120000])
#plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15.0, 17.5])
plt.xlabel("Number of Simulations per Step")
plt.ylabel("Discounted Return")
plt.title("NChain 95% CI, 50 seeds")


plt.plot(x, mean_w_125_300_15, "navy", label="W-MCTS-OS -std=12.5 -C=30.0 -p=1.5", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_125_300_15 - ci_125_300_15), (mean_w_125_300_15+ ci_125_300_15), color='navy', alpha = 0.1)

plt.plot(x, mean_150_50_15, "blue", label="W-MCTS-OS -std=15.0 -C=5.0 -p=1.5", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_150_50_15 - ci_150_50_15), (mean_150_50_15+ ci_150_50_15), color='blue', alpha = 0.1)

plt.plot(x, mean_150_300_15, "royalblue", label="W-MCTS-OS -std=15.0 -C=30.0 -p=1.5", linestyle= 'dotted',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_150_300_15- ci_150_300_15), (mean_150_300_15+ ci_150_300_15), color='royalblue', alpha = 0.1)

plt.plot(x, mean_1250_15, "mediumaquamarine", label="W-MCTS-TS -std=12.5 -p=1.5", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_1250_15 - ci_1250_15), (mean_1250_15+ ci_1250_15), color='mediumaquamarine', alpha = 0.1)

plt.plot(x, mean_15_15, "teal", label="W-MCTS-TS -std=15.0 -p=1.5", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_15_15 - ci_15_15), (mean_15_15+ ci_15_15), color='teal', alpha = 0.1)

plt.plot(x, mean_175_15, "turquoise", label="W-MCTS-TS -std=17.5 -p=1.5", linestyle= 'dotted',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_175_15 - ci_175_15), (mean_175_15+ ci_175_15), color='turquoise', alpha = 0.1)

plt.plot(x, mean_p_50_15, "maroon", label="Power-UCT -C=5.0 -p=1.5", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_p_50_15 - ci_p_50_15), (mean_p_50_15+ ci_p_50_15), color='maroon', alpha = 0.1)

plt.plot(x, mean_p_200_15, "red", label="Power-UCT -C=20.0 -p=1.5", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_p_200_15 - ci_p_200_15), (mean_p_200_15+ ci_p_200_15), color='red', alpha = 0.1)

plt.plot(x, mean_p_300_15, "indianred", label="Power-UCT -C=30.0 -p=1.5", linestyle= 'dotted',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_p_300_15 - ci_p_300_15), (mean_p_300_15+ ci_p_300_15), color='indianred', alpha = 0.1)

plt.ylim(bottom=1.0, top=6.0)
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.savefig('nchain_res.pdf')
plt.clf()



mean_w_10_10_15 = np.array([0.07, 0.07, 0.07, 0.11, 0.14, 0.23, 0.16, 0.38, 0.27, 0.24, 0.27, 0.44, 0.52, 0.48, 0.27, 0.62, 0.57, 0.37])
ci_10_10_15 = np.array([0.003, 0.004, 0.006, 0.008, 0.02, 0.09, 0.01, 0.13, 0.05, 0.05, 0.102, 0.25, 0.31, 0.26, 0.05, 0.037, 0.37, 0.1])

mean_075_20_15 = np.array([0.07, 0.07, 0.06, 0.1, 0.12, 0.23, 0.29, 0.3, 0.33, 0.68, 0.31, 0.43, 0.85, 0.4, 0.68, 2.1, 1.4, 2.3])
ci_075_20_15 = np.array([0.003, 0.005, 0.005, 0.004, 0.005, 0.09, 0.1, 0.07, 0.06, 0.18, 0.06, 0.1, 0.29, 0.1, 0.13, 0.58, 0.36, 0.59])

mean_075_300_15 = np.array([0.07, 0.07, 0.07, 0.09, 0.12, 0.32, 0.18, 0.16, 0.76, 0.59, 0.63, 1.02, 1.19, 0.21, 0.54, 1.22, 0.15, 0.75 ])
ci_075_300_15 = np.array([0.003, 0.004, 0.005, 0.005, 0.009, 0.18, 0.03, 0.02, 0.36, 0.2, 0.16, 0.25, 0.35, 0.05, 0.33, 0.45, 0.0003, 0.43])

mean_05_15 = np.array([0.07, 0.06, 0.07, 0.08, 0.11, 0.18, 0.21, 0.17,0.15, 0.15, 0.19, 0.15, 0.16, 0.17, 0.52, 1.63, 2.7, 3.3])
ci_05_15= np.array([0.01, 0.004, 0.01, 0.003, 0.006, 0.03, 0.05, 0.03, 0.004, 0.006, 0.04, 0.0008, 0.01, 0.01, 0.37, 0.59, 0.49, 0.68])

mean_075_15 = np.array([0.06, 0.06, 0.07, 0.07, 0.1, 0.12, 0.18, 0.13, 0.14, 0.14, 0.24, 0.15, 0.16, 0.15, 0.15, 0.14, 2.1, 3.2])
ci_075_15= np.array([0.007, 0.004, 0.004, 0.003, 0.004, 0.005, 0.04, 0.002, 0.002, 0.001, 0.095, 0.0007, 0.01, 0.0004, 0.00039, 0.001, 0.39, 0.68])

mean_1250_15 = np.array([0.07, 0.06, 0.07, 0.07, 0.1, 0.25, 0.12, 0.23, 0.14, 0.22, 0.18, 0.15, 0.15, 0.15, 0.15, 0.15, 1.88, 2.77])
ci_1250_15= np.array([0.01, 0.003, 0.005, 0.004, 0.004, 0.09, 0.003, 0.09, 0.001, 0.06, 0.02, 0.0008, 0.0003, 0.0001, 0.0001, 0.0005, 0.7, 0.56])

mean_p_075_15 = np.array([0.07, 0.07, 0.07, 0.07, 0.11, 0.12, 0.12, 0.15, 0.14, 0.14, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
ci_p_075_15= np.array([0.003, 0.004, 0.005, 0.005, 0.004, 0.004, 0.004, 0.01, 0.001, 0.005, 0.001, 0.001, 0.0009, 0.0008, 0.0002, 0.00002, 0.0008, 0.0001])

mean_p_125_15 = np.array([0.08, 0.08, 0.06, 0.07, 0.11, 0.12, 0.12, 0.13, 0.14, 0.12, 0.14, 0.15, 0.15, 0.14, 0.15, 0.15, 0.14, 0.15])
ci_p_125_15= np.array([0.01, 0.008, 0.004, 0.007, 0.005, 0.003, 0.007, 0.002, 0.002, 0.003, 0.001, 0.001, 0.001, 0.001, 0.0001, 0.0003, 0.002, 0.0005])

mean_p_450_15 = np.array([0.07, 0.07, 0.06, 0.09, 0.12, 0.16, 0.12, 0.12, 0.12, 0.16, 0.14, 0.13, 0.14, 0.13, 0.15, 0.15, 0.004, 0.13])
ci_p_450_15= np.array([0.003, 0.005, 0.004, 0.01, 0.006, 0.03, 0.005, 0.002, 0.002, 0.02, 0.005, 0.002, 0.001, 0.003, 0.0003, 0.0006, 0.001, 0.002])

x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
plt.xticks([20000, 40000, 60000, 80000, 100000, 120000])
#plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15.0, 17.5])
plt.xlabel("Number of Simulations per Step")
plt.ylabel("Discounted Return")
plt.title("SixArms 95% CI, 50 seeds")


plt.plot(x, mean_w_10_10_15, "navy", label="W-MCTS-OS -std=1.0 -C=1.0 -p=1.5", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_10_10_15 - ci_10_10_15), (mean_w_10_10_15+ ci_10_10_15), color='navy', alpha = 0.1)

plt.plot(x, mean_075_20_15, "royalblue", label="W-MCTS-OS -std=0.75 -C=2.0 -p=1.5", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_075_20_15- ci_075_20_15), (mean_075_20_15+ ci_075_20_15), color='royalblue', alpha = 0.1)

plt.plot(x, mean_075_300_15, "blue", label="W-MCTS-OS -std=0.75 -C=30.0 -p=1.5", linestyle= 'dotted',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_075_300_15 - ci_075_300_15), (mean_075_300_15+ ci_075_300_15), color='blue', alpha = 0.1)

plt.plot(x, mean_05_15, "turquoise", label="W-MCTS-TS -std=0.5 -p=1.5", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_05_15 - ci_05_15), (mean_05_15+ ci_05_15), color='turquoise', alpha = 0.1)

plt.plot(x, mean_075_15, "teal", label="W-MCTS-TS -std=0.75 -p=1.5", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_075_15 - ci_075_15), (mean_075_15+ ci_075_15), color='teal', alpha = 0.1)

plt.plot(x, mean_1250_15, "mediumaquamarine", label="W-MCTS-TS -std=1.25 -p=1.5", linestyle= 'dotted',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_1250_15 - ci_1250_15), (mean_1250_15+ ci_1250_15), color='mediumaquamarine', alpha = 0.1)

plt.plot(x, mean_p_075_15, "red", label="Power-UCT -C=0.75 -p=1.5", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_p_075_15 - ci_p_075_15), (mean_p_075_15+ ci_p_075_15), color='red', alpha = 0.1)

plt.plot(x, mean_p_125_15, "maroon", label="Power-UCT -C=1.25 -p=1.5", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_p_125_15 - ci_p_125_15), (mean_p_125_15+ ci_p_125_15), color='maroon', alpha = 0.1)

plt.plot(x, mean_p_450_15, "indianred", label="Power-UCT -C=45.0 -p=1.5", linestyle= 'dotted',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_p_450_15 - ci_p_450_15), (mean_p_450_15+ ci_p_450_15), color='indianred', alpha = 0.1)

plt.ylim(bottom=0.0, top=3.8)
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.savefig('arms_res.pdf')
plt.clf()



mean_w_05_141_15 = np.array([0.38, 0.42, 0.54, 0.48, 0.56, 0.58])
ci_05_141_15 = np.array([0.135, 0.137, 0.138, 0.138, 0.138, 0.137])

mean_05_15 = np.array([0.6, 0.72, 0.76, 0.8, 0.68, 0.74])
ci_05_15= np.array([0.136, 0.124, 0.118, 0.111, 0.129, 0.122])

mean_p_075_15 = np.array([0.24, 0.4, 0.38, 0.6, 0.58, 0.58])
ci_p_075_15= np.array([0.118, 0.136, 0.135, 0.136, 0.137, 0.137])

x = np.array([5000, 10000, 20000, 40000, 80000, 120000])
plt.xticks([20000, 40000, 60000, 80000, 100000, 120000])
#plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15.0, 17.5])
plt.xlabel("Number of Simulations per Step")
plt.ylabel("Success Rate")
plt.title("FrozenLake 95% CI, 50 seeds")


plt.plot(x, mean_w_05_141_15, "navy", label="W-MCTS-OS -std=0.5 -C=1.4142 -p=1.5", path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_05_141_15 - ci_05_141_15), (mean_w_05_141_15+ ci_05_141_15), color='navy', alpha = 0.1)

plt.plot(x, mean_05_15, "turquoise", label="W-MCTS-TS -std=0.5 -p=1.5", path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_05_15 - ci_05_15), (mean_05_15+ ci_05_15), color='turquoise', alpha = 0.1)

plt.plot(x, mean_p_075_15, "red", label="Power-UCT -C=1.4142 -p=1.5", path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_p_075_15 - ci_p_075_15), (mean_p_075_15+ ci_p_075_15), color='red', alpha = 0.1)


plt.ylim(bottom=0.0, top=1.0)
plt.legend()
#plt.xscale('log')
plt.grid(True)
plt.savefig('frozenlake_res.pdf')
plt.clf()


mean_w_05_141_15 = np.array([0.08, 0.09333, 0.0933, 0.09333, 0.1, 0.1])
ci_05_141_15 = np.array([0.0248, 0.0331, 0.0331, 0.0331,0.0379, 0.0379 ])

mean_05_15 = np.array([0.64, 0.82, 0.64, 0.82, 0.9, 0.9])
ci_05_15= np.array([0.274, 0.216, 0.274, 0.216, 0.174, 0.174])

mean_p_075_15 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ci_p_075_15= np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

x = np.array([20000, 40000, 60000, 80000, 100000, 120000])
plt.xticks([20000, 40000, 60000, 80000, 100000, 120000])
#plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15.0, 17.5])
plt.xlabel("Number of Simulations per Step")
plt.ylabel("Average Undiscounted Reward")
plt.title("Taxi 95% CI, 10 seeds")


plt.plot(x, mean_w_05_141_15, "navy", label="W-MCTS-OS -std=0.5 -C=1.4142 -p=1.5", path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_05_141_15 - ci_05_141_15), (mean_w_05_141_15+ ci_05_141_15), color='navy', alpha = 0.1)

plt.plot(x, mean_05_15, "turquoise", label="W-MCTS-TS -std=0.5 -p=1.5", path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_05_15 - ci_05_15), (mean_05_15+ ci_05_15), color='turquoise', alpha = 0.1)

plt.plot(x, mean_p_075_15, "red", label="Power-UCT -C=1.4142 -p=1.5", path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_p_075_15 - ci_p_075_15), (mean_p_075_15+ ci_p_075_15), color='red', alpha = 0.1)


plt.ylim(bottom=0.0, top=1.2)
plt.legend()
#plt.xscale('log')
plt.grid(True)
plt.savefig('taxi_res.pdf')
plt.clf()




mean_w_075_12_500 = np.array([0.07, 0.07, 0.09, 0.08, 0.05, 0.23, 0.46, 0.56, 0.64, 0.58, 0.67, 0.52, 0.63, 0.72, 0.60, 0.51, 0.56, 0.61])
ci_075_12_500 = np.array([0.02, 0.01, 0.02, 0.02, 0.03, 0.04, 0.04, 0.05, 0.06, 0.06, 0.04, 0.06, 0.07, 0.07, 0.06, 0.07, 0.07, 0.07])

mean_w_075_12_1000 = np.array([0.07, 0.07, 0.09, 0.08, 0.05, 0.23, 0.39, 0.68, 0.67, 0.63, 0.62, 0.58, 0.76, 0.62, 0.61, 0.69, 0.71, 0.62])
ci_075_12_1000 = np.array([0.02, 0.01, 0.03, 0.04, 0.02, 0.05, 0.08, 0.08, 0.06, 0.06, 0.05, 0.07, 0.08, 0.08, 0.07, 0.06, 0.06, 0.07])

mean_075_40 = np.array([0.1, 0.07, 0.12, 0.05, 0.07, 0.25, 0.55, 0.77, 0.86, 0.92, 0.72, 0.46, 0.33, 0.4, 0.64, 0.55, 0.58, 0.63])
ci_075_40= np.array([0.02, 0.03, 0.02, 0.06, 0.04, 0.05, 0.06, 0.07, 0.02, 0.044, 0.04, 0.05, 0.06,0.07, 0.08,0.07, 0.08, 0.08])

mean_075_500 = np.array([0.09, 0.07, 0.07, 0.04,0.04, 0.21, 0.51, 0.74, 0.89, 0.95, 0.69, 0.65, 0.58, 0.57, 0.5, 0.6, 0.59, 0.68])
ci_075_500= np.array([0.02, 0.01, 0.02, 0.01, 0.01, 0.04, 0.07, 0.08, 0.07, 0.06, 0.05, 0.08, 0.06, 0.06, 0.06, 0.05, 0.06, 0.07])

mean_075_1000 = np.array([0.02, 0.11, 0.04, 0.03, 0.08, 0.24, 0.34, 0.85, 0.85, 0.78, 0.7, 0.42, 0.55, 0.61, 0.65, 0.63, 0.62, 0.84])
ci_075_1000= np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.09, 0.09, 0.08, 0.06, 0.06, 0.1, 0.06, 0.04, 0.04, 0.06, 0.05, 0.06])

meanD2 = np.array([0.05, 0.08, 0.07, 0.04, 0.53, 0.69, 0.83, 1.038, 1.24, 1.44, 1.47, 1.37, 1.55, 1.48, 1.67, 1.35, 1.29, 1.36])
ci_D2 = np.array([0.02, 0.02, 0.02, 0.01, 0.05, 0.05, 0.04, 0.06, 0.06, 0.05, 0.06, 0.05, 0.08, 0.07, 0.07, 0.07, 0.08,  0.06])



x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
plt.xticks([20000, 40000, 60000, 80000, 100000, 120000])
#plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15.0, 17.5])
plt.xlabel("Number of Simulations per Step")
plt.ylabel("Discounted Return")
plt.title("Rocksample [7x8] 95% CI, no Knowledge, 50 seeds")


plt.plot(x, mean_w_075_12_500, "navy", label="W-MCTS-OS -std=0.75 -C=1.2 -p=50.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_075_12_500 - ci_075_12_500), (mean_w_075_12_500+ ci_075_12_500), color='navy', alpha = 0.1)


plt.plot(x, mean_w_075_12_1000, "blue", label="W-MCTS-OS -std=0.75 -C=1.2 -p=100.0", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_075_12_1000 - ci_075_12_1000), (mean_w_075_12_1000+ ci_075_12_1000), color='blue', alpha = 0.1)

plt.plot(x, mean_075_40, "turquoise", label="W-MCTS-TS -std=0.75 -p=4.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_075_40 - ci_075_40), (mean_075_40+ ci_075_40), color='turquoise', alpha = 0.1)

plt.plot(x, mean_075_500, "teal", label="W-MCTS-TS -std=0.75 -p=50.0", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_075_500 - ci_075_500), (mean_075_500+ ci_075_500), color='teal', alpha = 0.1)

plt.plot(x, mean_075_1000, "mediumaquamarine", label="W-MCTS-TS -std=0.75 -p=100.0", linestyle= 'dotted',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_075_1000 - ci_075_1000), (mean_075_1000+ ci_075_1000), color='mediumaquamarine', alpha = 0.1)

plt.plot(x, meanD2, "red", label="D2NG", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (meanD2 - ci_D2), (meanD2+ ci_D2), color='red', alpha = 0.1)



plt.ylim(bottom=0.0, top=1.8)
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.savefig('rocksample78know_res.pdf')
plt.clf()

mean_w_075_12_500 = np.array([0.875, 0.725, 0.825, 0.725, 0.775, 1.075, 2.025, 2.075, 2.25, 2.8, 2.55, 2.45, 2.675, 2.525, 2.4, 2.625, 2.675, 2.4 ])
ci_075_12_500 = np.array([0.13, 0.11, 0.11, 0.11,0.11, 0.11, 0.14, 0.18, 0.13, 0.13, 0.12, 0.14, 0.14, 0.14, 0.12, 0.12, 0.14, 0.15])

mean_w_075_12_1000 = np.array([0.975, 0.725, 0.825, 0.725, 0.775, 1.075, 2.025, 2.075, 2.25, 2.8, 2.55, 2.45, 2.675, 2.525, 2.4, 2.625, 2.675, 2.4])
ci_075_12_1000 = np.array([0.12, 0.13, 0.14, 0.1, 0.11, 0.14, 0.16, 0.17, 0.18, 0.14, 0.12, 0.14, 0.18, 0.15, 0.17, 0.15, 0.13, 0.16])

mean_075_40 = np.array([0.9, 0.85, 0.85, 0.75, 0.925, 1.325, 1.375, 2.3,2.3, 2.1, 1.77, 2.125, 2.3, 2.55, 1.725, 2.02, 2.02, 1.9])
ci_075_40= np.array([0.1, 0.1, 0.09, 0.08, 0.09, 0.13, 0.12, 0.16, 0.14, 0.16, 0.16, 0.2, 0.18, 0.17, 0.17, 0.18, 0.15, 0.16])

mean_075_500 = np.array([1.125, 0.775, 0.95, 0.775, 0.95, 0.975, 1.7, 2.175, 2.325, 2.275, 2, 1.9, 2.075, 2.1, 2.225, 2.275, 2.475, 2.05])
ci_075_500= np.array([0.12, 0.07, 0.08, 0.05, 0.07, 0.06, 0.014,0.1, 0.15, 0.16, 0.14, 0.15, 0.14, 0.14, 0.15, 0.15, 0.20, 0.15])

mean_075_1000 = np.array([0.975, 0.925, 0.9, 0.775, 0.65, 1.25, 1.65, 1.95, 2.225, 1.925, 2.05, 2.05, 1.95, 2.225, 2.1, 2.275, 2.2, 2.025])
ci_075_1000= np.array([0.1, 0.09, 0.08, 0.08, 0.07, 0.12, 0.15, 0.14, 0.13, 0.15, 0.17, 0.17, 0.17, 0.18, 0.15, 0.17, 0.19, 0.14])

meanD2 = np.array([0.925, 0.875, 0.925, 1.075, 2.2, 2.425, 2.6, 2.6, 2.3, 2.475, 2.2, 2.35, 2.125, 2.35, 2, 2.275, 2.225, 1.975])
ci_D2 = np.array([0.1, 0.1, 0.11, 0.10, 0.13, 0.15, 0.14 ,0.14, 0.1,0.1, 0.09, 0.09, 0.08, 0.08, 0.07, 0.08, 0.08, 0.08])



x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
plt.xticks([20000, 40000, 60000, 80000, 100000, 120000])
#plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15.0, 17.5])
plt.xlabel("Number of Simulations per Step")
plt.ylabel("Undiscounted Return")
plt.title("Rocksample [7x8] 95% CI, 50 seeds")


plt.plot(x, mean_w_075_12_500, "navy", label="W-MCTS-OS -std=0.75 -C=1.2 -p=50.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_075_12_500 - ci_075_12_500), (mean_w_075_12_500+ ci_075_12_500), color='navy', alpha = 0.1)


plt.plot(x, mean_w_075_12_1000, "blue", label="W-MCTS-OS -std=0.75 -C=1.2 -p=100.0", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_075_12_1000 - ci_075_12_1000), (mean_w_075_12_1000+ ci_075_12_1000), color='blue', alpha = 0.1)

plt.plot(x, mean_075_40, "turquoise", label="W-MCTS-TS -std=0.75 -p=4.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_075_40 - ci_075_40), (mean_075_40+ ci_075_40), color='turquoise', alpha = 0.1)

plt.plot(x, mean_075_500, "teal", label="W-MCTS-TS -std=0.75 -p=50.0", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_075_500 - ci_075_500), (mean_075_500+ ci_075_500), color='teal', alpha = 0.1)

plt.plot(x, mean_075_1000, "mediumaquamarine", label="W-MCTS-TS -std=0.75 -p=100.0", linestyle= 'dotted',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_075_1000 - ci_075_1000), (mean_075_1000+ ci_075_1000), color='mediumaquamarine', alpha = 0.1)

plt.plot(x, meanD2, "red", label="D2NG", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (meanD2 - ci_D2), (meanD2+ ci_D2), color='red', alpha = 0.1)


plt.ylim(bottom=0.0, top=3.2)
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.savefig('rocksample78_total.pdf')
plt.clf()


mean_w_10_40_40 = np.array([0.02, 0.05, 0.06, 0.07, 0.19, 0.37, 0.83, 1.5, 1.7, 1.88, 2.04, 1.97, 2.16, 2.18, 2.13, 2.06, 2.2, 2.12])
ci_10_40_40 = np.array([0.009, 0.02, 0.02, 0.02, 0.05, 0.08, 0.1, 0.11, 0.09, 0.07, 0.08, 0.06, 0.06, 0.07, 0.08, 0.06, 0.06, 0.06])

mean_w_20_40_40 = np.array([0.02, 0.05, 0.06, 0.07, 0.19, 0.42, 0.84, 1.42, 2.13, 2.1, 2.2, 2.24, 2.04, 1.94, 1.59, 1.53, 1.52, 1.53])
ci_20_40_40 = np.array([0.009, 0.02, 0.02, 0.02, 0.05, 0.08, 0.11, 0.12, 0.09, 0.1, 0.07, 0.08, 0.08, 0.09, 0.1, 0.1, 0.1, 0.11])

mean_w_10_40 = np.array([0.05, 0.09, 0.02, 0.05, 0.08, 0.48, 1.19, 2.09, 2.8, 3.03, 2.16, 1.8, 2.04, 2.5, 2.45, 2.58, 2.64, 2.5])
ci_10_40 = np.array([0.01, 0.02, 0.008, 0.02, 0.022, 0.09, 0.13, 0.14, 0.07, 0.06, 0.13, 0.14, 0.13, 0.12, 0.1, 0.09, 0.11, 0.12])

mean_20_40 = np.array([0.04, 0.05, 0.03, 0.04, 0.08, 0.4, 0.99, 2.1, 2.9, 3.09, 3.31, 3.08, 2.8, 1.9, 1.6, 1.16, 1.57, 2.11])
ci_20_40= np.array([0.01, 0.01, 0.01, 0.016, 0.027, 0.06, 0.14, 0.12, 0.07, 0.05, 0.03, 0.05, 0.1,0.1, 0.13, 0.13, 0.12, 0.09])

mean_20_160 = np.array([0.03,  0.03, 0.03, 0.05, 0.06, 0.244, 0.97, 2.22, 2.9, 2.97, 3.1, 2.6, 1.86, 1.63, 1.53, 1.7, 2.07, 2.12])
ci_20_160= np.array([0.01, 0.01, 0.17, 0.014, 0.02, 0.05, 0.14, 0.14, 0.08, 0.06, 0.04, 0.07, 0.103, 0.12, 0.12, 0.09, 0.09, 0.09])

mean_20_640 = np.array([0.1, 0.15, 0.11, 0.14, 0.16, 0.37, 1.15, 1.9, 2.3, 2.3, 2.1, 1.43, 0.52, 0.51, 0.72, 0.71, 1.08, 1.2])
ci_20_640= np.array([0.02, 0.03, 0.02, 0.02, 0.03, 0.07, 0.1, 0.11, 0.06, 0.06, 0.08, 0.1, 0.07, 0.07, 0.09, 0.09, 0.07, 0.07])


mean_30_40 = np.array([0.12, 0.11, 0.1, 0.05, 0.09, 0.29, 0.08, 1.5, 2.6, 2.9, 3.0, 3.06, 3.1, 3.1, 3.19, 3.0, 2.4, 1.93])
ci_30_40= np.array([0.03, 0.02, 0.02, 0.01, 0.02, 0.05, 0.08, 0.12, 0.07, 0.06, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.06, 0.08])


meanD2 = np.array([0.06, 0.05, 0.07, 0.04, 1.05, 1.54, 2.26, 2.5, 2.92, 3.2, 3.4, 3.45, 3.49, 3.48, 3.49, 3.52, 3.51, 3.51])
ci_D2 = np.array([0.02, 0.03, 0.02, 0.05, 0.1, 0.07, 0.1, 0.07, 0.09, 0.07, 0.06, 0.03, 0.06, 0.03, 0.02, 0.02, 0.02, 0.02])



x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
plt.xticks([20000, 40000, 60000, 80000, 100000, 120000])
#plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15.0, 17.5])
plt.xlabel("Number of Simulations per Step")
plt.ylabel("Discounted Return")
plt.title("Rocksample [8x8] 95% CI, 50 seeds")


plt.plot(x, mean_w_10_40_40, "blue", label="W-MCTS-OS -std=1.0 -C=4.0 -p=4.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_10_40_40 - ci_10_40_40), (mean_w_10_40_40+ ci_10_40_40), color='blue', alpha = 0.1)

plt.plot(x, mean_w_20_40_40, "navy", label="W-MCTS-OS -std=2.0 -C=4.0 -p=4.0", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_20_40_40 - ci_20_40_40), (mean_w_20_40_40+ ci_20_40_40), color='navy', alpha = 0.1)

plt.plot(x, mean_w_10_40, "turquoise", label="W-MCTS-TS -std=1.0 -p=4.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_10_40 - ci_10_40), (mean_w_10_40+ ci_10_40), color='turquoise', alpha = 0.1)

plt.plot(x, mean_20_40, "teal", label="W-MCTS-TS -std=2.0 -p=4.0", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_20_40 - ci_20_40), (mean_20_40+ ci_20_40), color='teal', alpha = 0.1)

plt.plot(x, mean_20_160, "mediumaquamarine", label="W-MCTS-TS -std=2.0 -p=16.0", linestyle= 'dotted',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_20_160 - ci_20_160), (mean_20_160+ ci_20_160), color='mediumaquamarine', alpha = 0.1)

plt.plot(x, mean_20_640, "gold", label="W-MCTS-TS -std=2.0 -p=32.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_20_640 - ci_20_640), (mean_20_640+ ci_20_640), color='gold', alpha = 0.1)


plt.plot(x, meanD2, "red", label="D2NG", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (meanD2 - ci_D2), (meanD2+ ci_D2), color='red', alpha = 0.1)



plt.ylim(bottom=0.0, top=4.0)
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.savefig('rocksample88_res.pdf')
plt.clf()




mean_w_075_12_500 = np.array([0.07, 0.07, 0.07, 0.09, 0.12, 0.29, 0.54, 0.65, 0.65, 0.75, 0.77, 0.8, 0.69, 0.74, 0.64, 0.65, 0.69, 0.71])
ci_075_12_500 = np.array([0.02, 0.01, 0.02, 0.01, 0.02, 0.05, 0.09, 0.1, 0.06, 0.06, 0.06, 0.07, 0.08, 0.05, 0.03, 0.05, 0.07, 0.08])

mean_w_075_12_1000 = np.array([0.07, 0.07, 0.07, 0.09, 0.12, 0.377, 0.45, 0.58, 0.71, 0.66, 0.7, 0.75, 0.64, 0.56, 0.71, 0.64, 0.64, 0.689])
ci_075_12_1000 = np.array([0.02, 0.01, 0.02, 0.01, 0.02, 0.04, 0.09, 0.09, 0.05, 0.06, 0.06, 0.07, 0.09, 0.05, 0.03, 0.05, 0.07, 0.07])

mean_075_40 = np.array([0.08, 0.09, 0.1, 0.08, 0.14, 0.29, 0.52, 0.91, 1.06, 1.04, 0.65, 0.68, 0.79, 0.98, 0.7, 0.67, 0.75, 0.72])
ci_075_40= np.array([0.02, 0.01, 0.02, 0.02, 0.02, 0.07, 0.09, 0.09, 0.07, 0.07, 0.09, 0.07, 0.06, 0.06, 0.06, 0.07, 0.07, 0.06])

mean_075_500 = np.array([0.12, 0.1, 0.08, 0.05, 0.08, 0.27, 0.66, 0.82, 1.02, 0.97, 0.74, 0.75, 0.72, 0.78, 0.89, 0.93, 0.96, 0.81])
ci_075_500= np.array([0.02, 0.01, 0.02, 0.01, 0.02, 0.05, 0.09, 0.1, 0.06, 0.06, 0.06, 0.07, 0.08, 0.05, 0.03, 0.05, 0.07, 0.08])

mean_075_1000 = np.array([0.07, 0.07, 0.08, 0.08, 0.08, 0.3, 0.58, 0.82, 0.97, 0.83, 0.66, 0.79, 0.68, 0.77, 0.78, 0.91, 0.82, 0.9])
ci_075_1000= np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.09, 0.09, 0.08, 0.06, 0.06, 0.1, 0.06, 0.04, 0.04, 0.06, 0.05, 0.06])

meanD2 = np.array([0.09, 0.03, 0.08, 0.08, 0.57, 0.76, 0.96, 1.2, 1.34, 1.52, 1.42, 1.59, 1.44, 1.62, 1.36, 1.57, 1.57, 1.38])
ci_D2 = np.array([0.02, 0.008, 0.032, 0.02, 0.06, 0.07, 0.05, 0.06, 0.05, 0.04, 0.05, 0.06, 0.05,0.06, 0.06, 0.07, 0.08, 0.07])



x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
plt.xticks([20000, 40000, 60000, 80000, 100000, 120000])
#plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15.0, 17.5])
plt.xlabel("Number of Simulations per Step")
plt.ylabel("Discounted Return")
plt.title("Rocksample [7x8], 95% CI, 50 seeds")


plt.plot(x, mean_w_075_12_500, "navy", label="W-MCTS-OS -std=0.75 -C=1,2 -p=50.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_075_12_500 - ci_075_12_500), (mean_w_075_12_500+ ci_075_12_500), color='navy', alpha = 0.1)


plt.plot(x, mean_w_075_12_1000, "blue", label="W-MCTS-OS -std=0.75 -C=1,2 -p=100.0", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_075_12_1000 - ci_075_12_1000), (mean_w_075_12_1000+ ci_075_12_1000), color='blue', alpha = 0.1)

plt.plot(x, mean_075_40, "turquoise", label="W-MCTS-TS -std=0.75 -p=4.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_075_40 - ci_075_40), (mean_075_40+ ci_075_40), color='turquoise', alpha = 0.1)

plt.plot(x, mean_075_500, "teal", label="W-MCTS-TS -std=0.75 -p=50.0", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_075_500 - ci_075_500), (mean_075_500+ ci_075_500), color='teal', alpha = 0.1)

plt.plot(x, mean_075_1000, "mediumaquamarine", label="W-MCTS-TS -std=0.75 -p=100.0", linestyle= 'dotted',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_075_1000 - ci_075_1000), (mean_075_1000+ ci_075_1000), color='mediumaquamarine', alpha = 0.1)

plt.plot(x, meanD2, "red", label="D2NG", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (meanD2 - ci_D2), (meanD2+ ci_D2), color='red', alpha = 0.1)




plt.ylim(bottom=0.0, top=2.0)
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.savefig('rocksample78_res.pdf')
plt.clf()


mean_30_80 = np.array([0.14,0.13, 0.06, 0.08, 0.09, 0.28, 0.71, 1.76, 2.53, 2.82, 2.95, 3.09, 3.06, 2.96, 2.56, 2.09,1.82, 1.82])
ci_30_80= np.array([0.02, 0.02, 0.01, 0.02, 0.02, 0.05, 0.09, 0.12, 0.07, 0.04, 0.04, 0.03, 0.04, 0.05, 0.05, 0.07, 0.07, 0.09])

mean_20_40 = np.array([0.14, 0.07, 0.05, 0.06, 0.12, 0.26, 0.62, 1.67, 2.61, 2.91, 3.08, 3.02, 2.8, 2.1, 1.2, 0.74, 0.8, 1.01])
ci_20_40= np.array([0.03, 0.02, 0.01, 0.01, 0.02, 0.05, 0.08, 0.12, 0.05, 0.05, 0.05, 0.03, 0.04, 0.07, 0.1, 0.09, 0.1, 0.11])

mean_20_80 = np.array([0.08, 0.13, 0.1, 0.09, 0.06, 0.031, 0.61, 1.9, 2.6, 3.0, 3.04, 2.6, 1.9, 1.11, 0.82, 1.2, 1.4, 1.7])
ci_20_80= np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.06, 0.09, 0.11, 0.068, 0.04, 0.04, 0.05, 0.07, 0.09, 0.09, 0.09, 0.08, 0.07])

meanD2 = np.array([0.09, 0.1, 0.09, 0.1, 1.1, 1.5, 1.8, 2.3, 2.6, 2.97, 3.13, 3.3, 3.4, 3.51, 3.56, 3.6, 3.68, 3.8])
ci_D2 = np.array([0.01, 0.02, 0.02, 0.02, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.04, 0.05, 0.05, 0.06, 0.06, 0.06, 0.05, 0.05])



x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
plt.xticks([20000, 40000, 60000, 80000, 100000, 120000])
#plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15.0, 17.5])
plt.xlabel("Number of Simulations per Step")
plt.ylabel("Discounted Return")
plt.title("Rocksample [11x11], 95% CI, 50 seeds")



plt.plot(x, mean_20_40, "turquoise", label="W-MCTS-TS -std=1.0 -p=4.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_20_40 - ci_20_40), (mean_20_40+ ci_20_40), color='turquoise', alpha = 0.1)

plt.plot(x, mean_20_80, "teal", label="W-MCTS-TS -std=2.0 -p=8.0", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_20_80 - ci_20_80), (mean_20_80+ ci_20_80), color='teal', alpha = 0.1)

plt.plot(x, mean_30_80, "mediumaquamarine", label="W-MCTS-TS -std=2.0 -p=16.0", linestyle= 'dotted',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_30_80 - ci_30_80), (mean_30_80+ ci_30_80), color='mediumaquamarine', alpha = 0.1)


plt.plot(x, meanD2, "red", label="D2NG", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (meanD2 - ci_D2), (meanD2+ ci_D2), color='red', alpha = 0.1)



plt.ylim(bottom=0.0, top=4.0)
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.savefig('rocksample1111.pdf')
plt.clf()




mean_w_01_C015_20 = np.array([6.4, 7.6, 7.2, 8.2, 7.7, 8.5, 8.2, 8.9, 8.56, 6.74, 8.7, 9.5, 8.6, 8.7, 9.2, 9.5, 8.2])
ci_01_C015_20 = np.array([0.6, 0.6, 0.54, 0.49, 0.57, 0.45, 0.5, 0.5, 0.49, 0.5, 0.58, 0.59, 0.52, 0.56, 0.5, 0.48, 0.55])


mean_100_20 = np.array([7.4, 7.1, 7.8, 7.2, 7.9, 7.7, 9.05, 9.4, 9.6, 9.7, 10.6, 10.5, 11.17, 11.67, 11.45, 11.48, 11.62])
ci_100_20= np.array([0.61, 0.52, 0.5, 0.02, 0.54, 0.5, 0.45, 0.51, 0.41, 0.5, 0.5, 0.4, 0.35, 0.41, 0.35, 0.46, 0.36])



meanD2 = np.array([6.05, 7.6, 6.73, 7.5, 7.6, 8.2, 8.6, 10.7, 10.8, 10.2, 11.7, 11.6, 12.9, 13.14, 13.61, 13.975, 13.9])
ci_D2 = np.array([0.72, 0.6, 0.7, 0.68, 0.68, 0.63, 0.42, 0.47, 0.57, 0.41, 0.5, 0.4, 0.38, 0.35, 0.35, 0.4, 0.5])



x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
plt.xticks([20000, 40000, 60000])
#plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15.0, 17.5])
plt.xlabel("Number of Simulations per Step")
plt.ylabel("Discounted Return")
plt.title("Pocman [7x7] 95% CI, 50 seeds")


plt.plot(x, mean_w_01_C015_20, "navy", label="W-MCTS-OS -std=5.0 -C=12.0 -p=2.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_01_C015_20 - ci_01_C015_20), (mean_w_01_C015_20+ ci_01_C015_20), color='navy', alpha = 0.1)


plt.plot(x, mean_100_20, "turquoise", label="W-MCTS-TS -std=10.0 -p=2.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_100_20 - ci_100_20), (mean_100_20+ ci_100_20), color='turquoise', alpha = 0.1)



plt.plot(x, meanD2, "red", label="D2NG", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (meanD2 - ci_D2), (meanD2+ ci_D2), color='red', alpha = 0.1)



plt.ylim(bottom=5.0, top=14.5)
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.savefig('pocman77_res.pdf')
plt.clf()



mean_01_20 = np.array([6, 6.8, 6.1, 6.8, 7.1, 7.15, 8.31, 9.02, 8.9, 8.9, 8.8, 9.05, 9.11, 8.65, 9.47, 8.7, 9.3])
ci_01_20= np.array([0.6, 0.5, 0.54, 0.65, 0.45, 0.5, 0.47, 0.35, 0.56, 0.45, 0.5, 0.62, 0.45, 0.5, 0.62, 0.63, 0.6])


mean_005_20 = np.array([7.9, 7.4, 7.2, 7.1, 6.5, 8.1, 8.1, 8.9, 8.6, 9.4, 9.1, 8.4, 9.7, 9.5, 9.3, 10.2, 10.5])
ci_005_20= np.array([0.61, 0.64, 0.58, 0.53, 0.56, 0.47, 0.42, 0.45, 0.5, 0.39, 0.46, 0.51, 0.43, 0.41, 0.45, 0.42, 0.45])


meanD2 = np.array([6.28, 7.3, 7.14, 6.5, 7.2, 8.6, 7.2, 8.2, 9.2, 9.86, 10.15, 11.21, 11.6, 11.7, 12.8, 13.1, 13.9])
ci_D2 = np.array([0.5, 0.6, 0.56, 0.56, 0.4, 0.56, 0.62, 0.4, 0.56, 0.45, 0.57, 0.47, 0.5, 0.3, 0.4, 0.37, 0.36])



x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
plt.xticks([20000, 40000, 60000])
#plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15.0, 17.5])
plt.xlabel("Number of Simulations per Step")
plt.ylabel("Discounted Return")
plt.title("Pocman [7x7], no Knowledge, 95% CI, 50 seeds")


plt.plot(x, mean_01_20, "navy", label="W-MCTS-OS -std=5.0 -C=12.0 -p=2.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_01_20 - ci_01_20), (mean_01_20+ ci_01_20), color='navy', alpha = 0.1)

plt.plot(x, mean_005_20, "teal", label="W-MCTS-TS -std=10.0 -p=2.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_005_20 - ci_005_20), (mean_005_20+ ci_005_20), color='teal', alpha = 0.1)



plt.plot(x, meanD2, "red", label="D2NG", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (meanD2 - ci_D2), (meanD2+ ci_D2), color='red', alpha = 0.1)



plt.ylim(bottom=5.0, top=14.0)
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.savefig('pocman77know.pdf')
plt.clf()


mean_w_01_C015_20 = np.array([-0.02, -0.02, -0.03, -0.02, -0.02, -0.009, 0.00013, 0.011, 0.0089, 0.014, 0.018, 0.022, 0.022, 0.024, 0.024, 0.032, 0.027, 0.029])
ci_01_C015_20 = np.array([0.004, 0.003, 0.004, 0.004, 0.004, 0.003, 0.003, 0.003, 0.004, 0.003, 0.004, 0.003, 0.004, 0.003, 0.003, 0.004, 0.004,  0.007])


mean_w_005_C02_20 = np.array([-0.02, -0.02, -0.02, -0.01, -0.01, -0.01, -0.004, -0.001, 0.016, 0.002, 0.012, 0.023, 0.017, 0.023, 0.025, 0.029, 0.03, 0.03])
ci_005_C02_20 = np.array([0.004, 0.003, 0.004, 0.004, 0.003, 0.003, 0.004, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002])

mean_005_20 = np.array([-0.02, -0.02, -0.02, -0.03, -0.019, -0.011, -0.001, 0.0009, 0.013, 0.017, 0.021, 0.035, 0.033, 0.03, 0.049, 0.044, 0.048, 0.047])
ci_005_20= np.array([0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.003, 0.005, 0.003, 0.003, 0.004, 0.004, 0.004, 0.004, 0.004, 0.005, 0.003,  0.007])

mean_01_20 = np.array([-0.03, -0.03, -0.02, -0.02, -0.019, -0.017, -0.01, 0.008, 0.017, 0.015, 0.024, 0.02, 0.036, 0.036, 0.044, 0.044, 0.05, 0.06])
ci_01_20= np.array([-0.005, 0.004, 0.004, 0.004, 0.004, 0.005, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.003, 0.005, 0.004, 0.004, 0.003, 0.005])

meanD2 = np.array([-0.02, -0.02, -0.01, -0.02, -0.01, -0.005, 0.003, 0.014, 0.017, 0.022, 0.037, 0.034, 0.042, 0.035, 0.045, 0.057, 0.055, 0.08])
ci_D2 = np.array([0.0044, 0.004, 0.003, 0.003, 0.004, 0.004, 0.003, 0.004, 0.003, 0.004, 0.003, 0.003, 0.005, 0.003, 0.003, 0.005, 0.005, 0.01])



x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
plt.xticks([20000, 40000, 60000, 80000, 10000, 120000])
#plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15.0, 17.5])
plt.xlabel("Number of Simulations per Step")
plt.ylabel("Discounted Return")
plt.title("Pocman [7x7] 95% CI, 50 seeds")


plt.plot(x, mean_w_01_C015_20, "navy", label="W-MCTS-OS -std=0.1 -C=0.15 -p=2.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_01_C015_20 - ci_01_C015_20), (mean_w_01_C015_20+ ci_01_C015_20), color='navy', alpha = 0.1)

plt.plot(x, mean_w_005_C02_20, "royalblue", label="W-MCTS-OS -std=0.05 -C=0.2 -p=2.0", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_w_005_C02_20 - ci_005_C02_20), (mean_w_005_C02_20+ ci_005_C02_20), color='royalblue', alpha = 0.1)

plt.plot(x, mean_005_20, "turquoise", label="W-MCTS-TS -std=0.05 -p=2.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_005_20 - ci_005_20), (mean_005_20+ ci_005_20), color='turquoise', alpha = 0.1)

plt.plot(x, mean_01_20, "teal", label="W-MCTS-TS -std=0.1 -p=2.0", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_01_20 - ci_01_20), (mean_01_20+ ci_01_20), color='teal', alpha = 0.1)

plt.plot(x, meanD2, "red", label="D2NG", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (meanD2 - ci_D2), (meanD2+ ci_D2), color='red', alpha = 0.1)



plt.ylim(bottom=-0.1, top=0.15)
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.savefig('pocman77_res_2.pdf')
plt.clf()





mean_005_20 = np.array([-0.13, -0.14, -0.14, -0.12, -0.1, -0.08, -0.07, -0.06, -0.06, -0.05, -0.04, -0.03, -0.04, -0.02, -0.02, -0.014, -0.014, -0.015])
ci_005_20= np.array([0.005, 0.006, 0.005, 0.005, 0.005, 0.005, 0.005,  0.005, 0.005, 0.005, 0.004, 0.004, 0.006, 0.005, 0.005, 0.006, 0.006, 0.006])

mean_01_20 = np.array([-0.13, -0.13, -0.16, -0.12, -0.1, -0.09, -0.07, -0.05, -0.044, -0.05, -0.04, -0.04, -0.04, -0.04, -0.04, -0.03, -0.03, -0.04 ])
ci_01_20= np.array([0.006, 0.006, 0.005, 0.005, 0.005, 0.005, 0.006, 0.005, 0.005, 0.005, 0.005, 0.004, 0.006, 0.005, 0.005, 0.006, 0.006, 0.006])

mean_02_20 = np.array([-0.14, -0.13, -0.14, -0.12, -0.11, -0.07, -0.07, -0.06, -0.05, -0.05, -0.05, -0.05, -0.04, -0.04, -0.05, -0.05, -0.05, -0.05 ])
ci_02_20= np.array([0.006, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.007,  0.004, 0.005, 0.004, 0.005, 0.005, 0.005, 0.005, 0.006, 0.005, 0.005])

meanD2 = np.array([-0.15, -0.16, -0.1, -0.1, -0.08, -0.07, -0.06, -0.06, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.04, -0.05, -0.06, -0.06])
ci_D2 = np.array([0.005, 0.006, 0.004, 0.006, 0.004, 0.005, 0.005, 0.004, 0.004, 0.005, 0.004, 0.005, 0.0049, 0.0058, 0.0053, 0.004, 0.004, 0.005])



x = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
plt.xticks([20000, 40000, 60000, 80000, 100000, 120000])
#plt.yticks([0, 2.5, 5, 7.5, 10, 12.5, 15.0, 17.5])
plt.xlabel("Number of Simulations per Step")
plt.ylabel("Discounted Return")
plt.title("Pocman [7x7], no Knowledge, 95% CI, 50 seeds")



plt.plot(x, mean_005_20, "turquoise", label="W-MCTS-TS -std=0.05 -p=2.0", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_005_20 - ci_005_20), (mean_005_20+ ci_005_20), color='turquoise', alpha = 0.1)

plt.plot(x, mean_01_20, "teal", label="W-MCTS-TS -std=0.1 -p=2.0", linestyle= 'dashed',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_01_20 - ci_01_20), (mean_01_20+ ci_01_20), color='teal', alpha = 0.1)

plt.plot(x, mean_02_20, "mediumaquamarine", label="W-MCTS-TS -std=0.2 -p=2.0", linestyle= 'dotted',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (mean_02_20 - ci_02_20), (mean_02_20+ ci_02_20), color='mediumaquamarine', alpha = 0.1)

plt.plot(x, meanD2, "red", label="D2NG", linestyle= 'solid',path_effects=[pe.Stroke(linewidth=lw, foreground='black'), pe.Normal()])
plt.fill_between(x, (meanD2 - ci_D2), (meanD2+ ci_D2), color='red', alpha = 0.1)



plt.ylim(bottom=-0.25, top=0.1)
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.savefig('pocman77know_2.pdf')
plt.clf()



optWstein_visits = np.array([1.15306e+10, 1.12099e+10, 3.50043e+08, 2.31763e+08, 1.14824e+08, 6.88763e+07, 2.58069e+07 ])
optWstein_visits2 = np.array([2.6597e+08, 2.12622e+08, 7.14081e+06, 7.77417e+08, 5.27626e+08, 5.56541e+08, 6.21144e+06 ])
wstein_visits = np.array([4.00847e+08, 3.84308e+08, 1.18092e+07, 7.89136e+06, 3.98733e+06, 3.57054e+08, 1.18763e+09 ])
puct_visits = np.array([1.10968e+09, 1.1692e+09, 3.30443e+07, 2.19048e+07, 1.09477e+07, 6.56369e+06, 2.18869e+06 ])

optWstein_visits = optWstein_visits/sum(optWstein_visits)
optWstein_visits2 = optWstein_visits2/sum(optWstein_visits2)
wstein_visits = wstein_visits/sum(wstein_visits)
puct_visits = puct_visits / sum(puct_visits)

arms_shape = (1, 7)
optWstein_visits = optWstein_visits.reshape(arms_shape)
optWstein_visits2 = optWstein_visits2.reshape(arms_shape)
wstein_visits = wstein_visits.reshape(arms_shape)
puct_visits = puct_visits.reshape(arms_shape)

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 8),
                        subplot_kw={'xticks': [], 'yticks': []})

im0 = axs[0].imshow(optWstein_visits, interpolation='nearest', cmap='pink')
axs[0].set_title("W-MCTS-OS \n std=0.75 C=30.0")
im3 = axs[1].imshow(optWstein_visits2, interpolation='nearest', cmap='pink')
axs[1].set_title("W-MCTS-OS \n std=0.75 C=2.0")
im1 = axs[2].imshow(wstein_visits, interpolation='nearest', cmap='pink')
axs[2].set_title("W-MCTS-TS \n std=0.75")
im2 = axs[3].imshow(puct_visits, interpolation='nearest', cmap='pink')
axs[3].set_title("Power-UCT \n C=1.0")
cb0 = fig.colorbar(im0, ax=axs[0])
for t in cb0.ax.get_yticklabels():
    t.set_fontsize(20)
cb1 = fig.colorbar(im1, ax=axs[2])
for t in cb1.ax.get_yticklabels():
    t.set_fontsize(20)
cb1 = fig.colorbar(im2, ax=axs[3])
for t in cb1.ax.get_yticklabels():
    t.set_fontsize(20)
cb1 = fig.colorbar(im3, ax=axs[1])
for t in cb1.ax.get_yticklabels():
    t.set_fontsize(20)

fig.suptitle('Relative visits on SixArms', fontsize=20)
plt.savefig('arms_visits.pdf')


optWstein_visits = np.array([1.17763e+09, 5.86267e+08, 2.91713e+08, 1.45635e+08, 1.52281e+08  ])
wstein_visits = np.array([1.15984e+09, 5.77454e+08, 2.89217e+08, 1.46395e+08, 1.80621e+08  ])
puct_visits = np.array([1.17811e+09, 5.88316e+08, 2.93847e+08, 1.46854e+08, 1.46399e+08 ])


optWstein_visits = optWstein_visits/sum(optWstein_visits)
wstein_visits = wstein_visits/sum(wstein_visits)
puct_visits = puct_visits / sum(puct_visits)

arms_shape = (1, 5)
optWstein_visits = optWstein_visits.reshape(arms_shape)
wstein_visits = wstein_visits.reshape(arms_shape)
puct_visits = puct_visits.reshape(arms_shape)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 8),
                        subplot_kw={'xticks': [], 'yticks': []})

im0 = axs[0].imshow(optWstein_visits, interpolation='nearest', cmap='pink')
axs[0].set_title("W-MCTS-OS \n std=12.50 C=30.0")
im1 = axs[1].imshow(wstein_visits, interpolation='nearest', cmap='pink')
axs[1].set_title("W-MCTS-TS \n std=12.50")
im2 = axs[2].imshow(puct_visits, interpolation='nearest', cmap='pink')
axs[2].set_title("Power-UCT \n C=25.0")
cb0 = fig.colorbar(im0, ax=axs[0])
for t in cb0.ax.get_yticklabels():
    t.set_fontsize(20)
cb1 = fig.colorbar(im1, ax=axs[1])
for t in cb1.ax.get_yticklabels():
    t.set_fontsize(20)
cb1 = fig.colorbar(im2, ax=axs[2])
for t in cb1.ax.get_yticklabels():
    t.set_fontsize(20)

fig.suptitle('Relative visits on NChain', fontsize=20)
plt.savefig('chain_visits.pdf')

#1.5822e+09 4.67227e+08 1.77844e+08 9.8728e+07 2.75321e+07

optWstein_visits = np.array([1.58841e+09, 4.68775e+08, 1.75584e+08, 9.47532e+07, 2.60079e+07   ])
wstein_visits = np.array([1.66186e+09, 4.72124e+08, 1.504e+08, 5.64671e+07, 1.26781e+07 ])
puct_visits = np.array([1.66078e+09, 4.72332e+08, 1.5191e+08, 5.63079e+07, 1.22038e+07  ])


optWstein_visits = optWstein_visits/sum(optWstein_visits)
wstein_visits = wstein_visits/sum(wstein_visits)
puct_visits = puct_visits / sum(puct_visits)

arms_shape = (1, 5)
optWstein_visits = optWstein_visits.reshape(arms_shape)
wstein_visits = wstein_visits.reshape(arms_shape)
puct_visits = puct_visits.reshape(arms_shape)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 8),
                        subplot_kw={'xticks': [], 'yticks': []})

im0 = axs[0].imshow(optWstein_visits, interpolation='nearest', cmap='pink',norm=matplotlib.colors.Normalize())
axs[0].set_title("W-MCTS-OS \n std=7.50 C=7.5")
im1 = axs[1].imshow(wstein_visits, interpolation='nearest', cmap='pink', norm=matplotlib.colors.Normalize())
axs[1].set_title("W-MCTS-TS \n std=12.50")
im2 = axs[2].imshow(puct_visits, interpolation='nearest', cmap='pink', norm=matplotlib.colors.Normalize())
axs[2].set_title("Power-UCT \n C=7.5")
cb0 = fig.colorbar(im0, ax=axs[0])
for t in cb0.ax.get_yticklabels():
    t.set_fontsize(20)
cb1 = fig.colorbar(im1, ax=axs[1])
for t in cb1.ax.get_yticklabels():
    t.set_fontsize(20)
cb1 = fig.colorbar(im2, ax=axs[2])
for t in cb1.ax.get_yticklabels():
    t.set_fontsize(20)

fig.suptitle('Relative visits on RiverSwim', fontsize=20)
plt.savefig('river_visits.pdf')



puct = np.array([
[4225567, 2449852, 1643245,  609733, 2224137, 1160772,  491157,  219262,
  838434,  323611,  204929,   36383,  187595,  113563,   86977,   17512],
[1145863,  356919,  105577,   25984,  932778,  328793,   34507,    8273,
  570301,  250519,   76540,   14147,  140427,   91854,   48538,   10450],
[2952147, 1175382,  373697,  106170, 1527128,  670337,  106496,   32808,
  769742,  396378,  164190,   30238,  225818,  440982,  325970,   92820],
[2809248, 1236644,  400674,  118661, 1836594,  775693,  115822,   37496,
 1235611,  532944,  173939,   31245,  300158,  195837,  106806,   22728],
[1341513,  777640,  283542,   88336,  992660,  533867,  148416,   42706,
  942799,  717286,  396941,   69401,  299481,  631194,  599826,  172333],
[1980134,  747673,  237312,   66748, 1089613,  464034,   72558,   21207,
  618879,  306321,  126927,   23338,  182203,  276528,  251851,   71955],
[2670643,  839654,  244556,   61249, 1894557,  629877,   55612,   15056,
  847161,  216577,   58837,    8889,  189230,   64330,   27204,    4119],
[6.92048e+05, 2.20530e+05, 6.56060e+04, 1.70110e+04, 3.42553e+05, 1.24534e+05,
 1.33230e+04, 3.97300e+03, 1.07566e+05, 2.42120e+04, 6.86200e+03, 7.41000e+02,
 2.18750e+04, 6.02600e+03, 2.15500e+03, 1.72000e+02],
[2.74805e+05, 8.57780e+04, 2.46460e+04, 5.83300e+03, 2.05942e+05, 6.56400e+04,
 4.96500e+03, 1.30900e+03, 6.68220e+04, 1.59680e+04, 4.12600e+03, 5.75000e+02,
 1.40040e+04, 4.24700e+03 ,1.58300e+03, 1.72000e+02],
[4258954, 2491087, 1394015,  501320, 2228941, 1125639,  418711,  180330,
  817077,  215318,  150543,   25726,  174787,   64741,   51802,    9041],
[2002037, 1365673,  951349,  346519, 1684178, 1059901,  527175,  172204,
 1924461, 1556852, 1036283,  194873,  735341, 1914324, 1769910,  515155],
[831568, 650291, 668128, 263933, 271387, 291918, 288195, 114891, 118731,
 177609, 218459,  33589,  38418,  86804, 106671,  27753],
[6.92048e+05, 2.20530e+05, 6.56060e+04, 1.70110e+04, 3.42553e+05, 1.24534e+05,
 1.33230e+04, 3.97300e+03, 1.07566e+05, 2.42120e+04, 6.86200e+03, 7.41000e+02,
 2.18750e+04, 6.02600e+03, 2.15500e+03, 1.72000e+02],
[4748652, 1844221,  577147,  162217, 3341791, 1285300,  160499,   48844,
 2184912,  787149,  238940,   43083,  520940,  275719,  140071,   28094],
[1633647,  636010,  205901,   58221, 1092632,  438254,   76305,   22039,
  630860,  296683,  186253,   35406,  156391,  118481,  101537,   25430],
[674105, 203954,  64131,  14517, 717748, 303995,  43027,   9019, 994892,
 482412, 150386,  29542, 251901, 185753, 102757,  22508],
[ 784734,  242248,  100504,   23596,  941895,  457101,  136840,   29309,
  970964,  898001,  614445,  137175,  368620,  883667, 1008250,  296378],
[2825461, 2625794, 3250108, 2392626,  989354,  989440,  975909,  732021,
  294462,  118937,  275883,   50399,   56716,   44318,   86323,   16806],
[573865, 178876,  54466,  13199, 468357, 177640,  21693,   5218, 296399,
 178821,  57869,  10893,  77510,  70444,  39549,   9211],
[1874834,  706941,  654061,  389929, 2041258,  979544,  513815,  187672,
 2151929, 1530952,  924362,  163535,  696990, 1325783,  924780,  251450],
[1011851,  310188,   96298,   22197, 1021491,  378559,   58863,   12373,
  650204,  427434,  198006,   39247,  237313,  626592,  529349,  154991],
[2308503, 2092941,  901535,  322957,  842597,  701937,  295929,  120231,
  247878,   65532,   87036,   14677,   46773,   19717,   26533,    4969],
[510324, 161184,  55712,  14634, 335653, 150199,  39949,   9766, 265368,
 202950, 157659,  33423,  75015,  92401,  89287,  23367],
[1151118, 1030086, 1655965, 1452449,  353364,  416485,  583098,  449971,
  106290,   59174,  169130,   31455,   21326,   25744,   55673,   11774],
[935993, 301140, 107300,  27945, 509753, 274918,  81465,  19016, 429474,
 538570, 295142,  58086, 159860, 492115, 290947,  78695],
[2649875, 1300363,  799972,  286740, 2168997,  954617,  497409,  158063,
 1775400,  564219,  371593,   65521,  418256,  209144,  177233,   40352],
[1585934,  615673,  193991,   54277,  997489,  408388,   56022,   16983,
  698275,  289770,   91283,   17205,  168283,  106377,   56635,   11982],
[300823,  98660,  54103,  14892, 271694, 149446,  90196,  21426, 264674,
 263441, 341057,  62022,  90385, 167266, 415997, 124831],
[3433926, 1442440,  479728,  139873, 2078459,  958258,  189034,   54741,
 1764490,  986069,  437813,   85739,  493401,  640684,  365203,   92160],
[800461, 554639, 384445, 141069, 288559, 317009, 267979,  82770, 209581,
 447921, 452485,  85896, 119903, 534658, 597308, 175345]
])


opt = np.array([
[1743718,  860854,  427202,  255746, 1078739,  713369,  133199,  121203,
  507250,  167454,   85822,   22885,  170326,   72489,   53397,   15649,],
[1332071,  563467,  211014,   76002,  801926,  541835,  102543,   53932,
  417711,  269411,  231906,   70396,  243193,  422581,  500629,  164062,],
[1062909,  411727,  117047,   39815,  617985,  403688,   55899,   28168,
  320413,  219601,  116580,   33911,  239722,  460645,  242022,   78473,],
[2971663, 1036104,  316319,  118250, 2209117, 1200509,  122205,   69386,
 1073285,  416167,  176236,   44421,  389964,  215590,  141334,   43191,],
[1657595,  589662,  186159,   64774, 1104049,  642003,   69536,   39501,
  671549,  268353,   98507,   19952,  253373,  136136,   85373,   26257,],
[1413421,  548874,  158148,   48231,  924425,  533750,   71410,   34940,
  719662,  326254,  124345,   36002,  272198,  157353,  106706,   33163],
[2468774, 1294814,  501743,  210238, 1553700, 1061979,  191989,  121685,
  779229,  417237,  301052,   91689,  340050,  316255,  234227,   74926],
[802199, 331130,  86543,  27537, 517727, 381100,  56154,  24580, 370102,
 383553, 137688,  36948, 273215, 489742, 246766,  79919],
[270503, 109235,  33470,  12289, 261298, 215035,  51971,  19832, 305241,
 263940, 175449,  52947, 253991, 496891, 407484, 134439],
[2330048,  976243,  280217,   98971, 1080974,  676412,   83397,   52884,
  357895,   94764,   45807,   11545,  118147,   34910,   25753,    6824],
[883593, 375812, 138971,  35295, 522847, 421400, 182297,  68800, 471767,
 324622, 233017,  64138, 190349, 138798, 143723,  45754],
[532882, 184797,  61352,  17474, 482889, 350703, 114236,  41158, 323602,
 326504, 393661, 106216, 270295, 544882, 850965, 281132],
[891511, 350235,  84017,  27535, 544771, 348211,  33508,  17290, 391768,
 193218,  74725,  21131, 153590,  93516,  66081,  20428],
[1728612, 1042627,  696031,  205902,  817565,  772519,  462212,  213761,
  272747,  139710,  233966,   72914,  100382,   84267,  108444,   33827],
[10450512,  5438172,  2437287,   894202,  6800635,  4403948,  1115250,
   619122,  2655256,   973922,   943834,   244068,   963466,   496976,
   565603,   175763],
[3328591, 1409327,  586928,  234424, 1779792, 1287144,  329403,  169765,
 1069778,  671783,  432281,  128506,  516327,  596699,  503738,  162806],
[1498782,  702698,  373961,  152576,  989558,  562583,  196810,  109757,
  340418,  101859,   94447,   27177,  108939,   47375,   54057,   16091],
[6184579, 2783131, 1116142,  441033, 3496041, 2252635,  338781,  234071,
 1411448,  563568,  372227,  104957,  531205,  366121,  400632,  126311],
[496548, 202785,  65080,  20337, 469774, 276732,  26923,  13550, 300852,
 175858,  40278,  11502, 114239,  65832,  38696,  11937],
[5661855, 2306667,  808565,  327976, 4014952, 2258609,  267819,  176553,
 1797464,  639447,  259819,   68924,  638771,  280408,  192604,   57067],
[5589772,  849898,  413173,   19062, 4033105,  177862,  280873,   64849,
 2652011, 1438664,  569899,    9026,   29994,  993344,  780629,  233937],
[2926720,  344514,  193429,    7828, 2517305,  171021,  325563,   92828,
 1703421, 1191487,  629809,    5642,   32135, 1027072,  856988,  266211],
[1985397,  381558,  185703,    4950, 1680961,  175948,  191361,   47872,
 1623768, 1493923,  526269,   23455,   15593, 2125069, 1581526,  503605],
[5485219,  460266,  215618,    6347, 4524622,  358292,  182991,   40889,
 2723924, 1340150,  456795,   11583,   30677,  981417,  718439,  212895],
[3620158,  370550,  191901,    9877, 3046760,  194124,  236765,   60941,
 2589721, 1746813,  692990,   32862,   63588, 1588550, 1258437,  392022],
[3177669,  350939,  167793,    4080, 2694674,  161682,  147286,   35375,
 2362447, 1365193,  464930,   40283,   21126, 1080894,  797772,  245145],
[7789744, 1878618, 1019928,  103927, 7272972,  409871, 1203526,  296752,
 6838723, 5279913, 2429179,    9247,   76406, 5288908, 3995594, 1249968],
[3118204,  655211,  405060,   17832, 3559221,  329687,  802656,  203496,
 4137874, 3518933, 1691184,    8666,   12824, 3080025, 2181011,  678397],
[367464,  70689,  37045,    763, 346051, 186240,  34603,   8166, 314394,
 526274, 180338,  35458,  56320, 668835, 399257, 126566],
[5058176, 1372873,  854384,   26733, 3477849,  358914,  711059,  168207,
 2228739, 1099699,  721793,   16551,   18322,  697473,  599350,  176760],
[4367348, 1395725,  686609,    9842, 3878690,  310384,  575672,  153890,
 3178497, 2236561,  959759,    8412,   16345, 2064253, 1351620,  413814],
[1541199,  235537,  140670,    1836, 1477338,  105821,  294730,   70807,
 1338085,  995014,  564260,    4724,    8651,  778366,  726922,  227256],
[9.264096e+06, 2.420825e+06, 1.295860e+06, 3.630900e+04, 7.820440e+06,
 3.164420e+05, 7.732800e+05, 1.956890e+05, 5.490923e+06, 2.626869e+06,
 1.034634e+06, 8.630000e+03, 3.554000e+04, 1.568569e+06, 1.078413e+06,
 3.107320e+05],
[8235592, 2821852, 2173002,  125132, 8029056,  469650, 1540461,  382847,
 7139091, 4462810, 2110409,   24399,   27188, 3585875, 2771902,  850336],
[2273407,  458995,  228580,    3157, 2314002,  148267,  190960,   46696,
 2151635,  986384,  439380,    8621,  127608,  686074,  601640,  182927],
[6464863, 1500584,  754224,   36925, 6849796,  345605,  694400,  180699,
 6753384, 4323300, 1726831,   56323,   50329, 3362370, 2622259,  809276],
[1017051,  172966,   92905,    1975, 1163299,  252076,   71790,   14634,
 1197889,  606822,  216679,   20837,  190603,  359489,  243363,   70765],
[5941406,  676090,  376765,   16842, 5660649,  229284,  603901,  144904,
 5098930, 3307277, 1374336,    7251,   27083, 2858153, 2254180,  699995],
[1317560,  232705,  143384,    5942, 1562126,  205614,  359243,   94817,
 1838226, 2392586, 1030636,    6012,   40178, 3440358, 2931149,  940920],
[6781198, 1577762,  774940,   92962, 6028871,  675376,  693146,  195026,
 4568641, 3151385, 1356953,   32010,   96003, 2993747, 2298927,  712856],
[642616, 133582,  69512,   1342, 665094,  61539, 106993,  28703, 679911,
 612000, 309442,   9355,   6376, 719135, 951516, 307471],
[2300297,  423003,  215089,    6294, 3307568,  364100,  307836,   81384,
 2643181, 1336622,  586436,   12524,   57119,  836755,  605601,  178742],
[1221813,  115463,   60774,    1609,  981185,   39959,  101434,   26563,
  792576,  548403,  279882,    3273,    8699,  446313,  444740,  139889],
[5837218, 1010976,  511063,   23005, 6001715,  274280,  517937,  121568,
 5392821, 3683915, 1299284,   16749,   48563, 4356571, 3761663, 1192401],
[7306290, 2074418, 1277898,   34904, 6501349,  447089, 1137591,  322977,
 5386425, 3755034, 1922706,   14432,   45500, 3249023, 2685356,  835167],
[1218499,  210511,  115196,    2518, 1678855,  313163,   91862,   17900,
 1723833,  754049,  259340,   19908,  230508,  445656,  291156,   82669],
[2850459,  342869,  165903,    5258, 2324499,  150768,  105782,   21486,
 1763733,  904003,  387025,   58159,  159273,  548595,  392871,  113569],
[4262731,  519253,  243597,    5279, 4230011,  332539,  284726,   69864,
 3076280, 1901341,  754497,   12850,   73518, 1808740, 1376297,  425092],
[2.324148e+06, 2.034020e+05, 1.053060e+05, 1.687000e+03, 2.303948e+06,
 1.024910e+05, 1.258870e+05, 2.619700e+04, 1.946449e+06, 1.292749e+06,
 4.359700e+05, 3.498700e+04, 5.021400e+04, 1.273000e+06, 1.004162e+06,
 3.146920e+05],
[4771504,  781977,  447304,   32490, 4483375,  255700,  435284,  101103,
 3581316, 2234461, 1064181,   20193,   63319, 1560984, 1277869,  391371],
[2010539,  425098,  250290,   10474, 3049922,  267221,  455626,  123802,
 3682038, 3509960, 1256314,    6299,   63531, 4219525, 2779370,  878361],
[2951954,  360780,  179554,    7228, 2642700,  104188,  120567,   25051,
 1991919, 1041183,  428987,   60174,   23446,  720702,  612394,  185761],
[20751717,  3660370,  1964802,    36305, 18369250,   761176,  1345998,
   296345, 11342352,  5257475,  2141849,    30247,    98227,  3071525,
  2127660,   603349],
[3764995,  543414,  272247,    4731, 3948398,  192536,  320795,   76596,
 3597880, 2202720,  942481,   46844,   61181, 1518497, 1206469,  369120],
[4048982,  995702,  578317,   11909, 3950812,  332200,  648489,  168001,
 4069907, 2508386, 1151861,   41439,  111403, 1813449, 1313697,  399536],
[1857099,  401242,  204670,    2900, 1701102,  102792,  169159,   40640,
 1506303, 1210739,  427034,   18969,  101886, 1694230, 1449692,  462499],
[3046299,  494134,  249570,    4760, 2314154,  223019,  116988,   20679,
 1515619,  765552,  345976,   58990,   94569,  454967,  332582,   95877],
[1668320,  296490,  139501,    2610, 1620248,  175895,  114669,   24474,
 1155664,  793399,  280813,    9429,   27608,  873724,  635892,  198135],
[5690296, 1239561,  730729,   13047, 4848355,  214392,  434637,  113653,
 2769808, 1164415,  477439,    6424,   26932,  650476,  444318,  122722],
[5269226, 1503113,  955493,  108663, 4463515,  256996,  601100,  171171,
 3145742, 1905833,  853736,   31395,   22912, 1441351, 1115396,  337711],
[7.032829e+06, 3.267345e+06 ,2.632223e+06, 1.749400e+04, 5.614774e+06,
 4.710950e+05, 1.162438e+06, 2.767820e+05, 3.520658e+06 ,1.588812e+06,
 8.337570e+05 ,4.520000e+03 ,1.768900e+04, 9.343950e+05 ,7.190150e+05,
 2.048910e+05],
[ 866112,  159501,  116376,    9226, 1154683,  225803,  247355,   59789,
 1866495, 1614724,  750174,   23935,  242631, 1821582, 1904738,  610829],
[629990, 131332,  66996,   1319, 639106,  59315,  99314,  27244, 627561,
 516918, 277422,   6693,   6162, 502471, 595030, 190693],
[5649782, 1163437,  568219,    7800, 4555096,  330788,  344634,   78227,
 2882580, 1759223,  633816,    9257,   61218, 1740285, 1154825,  351333],
[5508977, 1065697,  516230,   29953, 4981738,  268670,  556968,  153409,
 3430158, 1706481,  846504,   11590,   51063, 1066628,  808353,  237186],
[6.260284e+06, 7.009830e+05, 3.330220e+05, 4.280000e+03, 5.512260e+06,
 1.581170e+05, 2.635700e+05, 6.184200e+04 ,3.630432e+06 ,1.804805e+06,
 6.229470e+05 ,1.296900e+04, 2.270100e+04 ,1.370553e+06, 1.085839e+06,
 3.277810e+05],
[2517978,  626927,  266256,    7150, 1827671,  161129,  110310,   22591,
 1058191,  415849,  151471,   10561,   76039,  205064,  128408,   33107],
[11050504,  3182524,  1581721,    72291,  8743850,   373632,   876527,
   213393,  6162271,  3879052,  1452092,    33952,    42675,  3711759,
  2396357,   729331],
[4.35506e+05, 7.94100e+04, 4.32170e+04, 8.66000e+02, 6.22250e+05, 2.82763e+05,
 4.20590e+04 ,9.30800e+03, 6.58109e+05, 7.71438e+05, 3.41433e+05, 7.99060e+04,
 1.33709e+05, 9.28850e+05 ,7.12678e+05, 2.26935e+05],
[3076147,  630962,  331024,   18955, 2965726,  128575,  246137,   51357,
 2339153, 1217113,  523099,   10909,   29355,  837847,  704791,  214297]
])

thompson = np.array([
    [672406, 693229, 631823, 242726, 247697, 449735, 518574, 172031, 214917,
     530358, 590283, 127076, 218447, 828543, 510925, 149297],
    [1456638, 464847, 144244, 38323, 878001, 365110, 61789, 17582,
     430817, 234040, 162296, 35395, 112337, 91308, 84590, 23067],
    [1144299, 366467, 122621, 31307, 606459, 361441, 88308, 21385,
     554749, 592811, 314061, 64242, 344661, 1034038, 804495, 240690],
    [3878595, 1223077, 389901, 99349, 3113983, 1275303, 225913, 59134,
     1995383, 1114695, 704716, 147547, 606429, 766757, 826038, 240367],
    [1728878, 549276, 179713, 46497, 1109451, 541357, 112168, 28992,
     944874, 671563, 365529, 76059, 330523, 560806, 537445, 158504],
    [8971040, 3416772, 1891776, 652723, 6967572, 2954519, 720562, 281317,
     4356889, 2088943, 757974, 146298, 1223908, 1209141, 634335, 167353],
    [2809438, 1013209, 314488, 87169, 1788814, 725411, 107177, 33602,
     725696, 323258, 217064, 44614, 184522, 150528, 226610, 65390],
    [1202781, 383705, 176917, 49436, 1151669, 719127, 260536, 65127,
     1607826, 1410571, 672945, 142096, 585566, 1025019, 593284, 167604],
    [314227, 98005, 41046, 10220, 319802, 183825, 56818, 13745, 410885,
     348828, 224177, 49097, 130535, 176065, 246209, 72441],
    [2947611, 951174, 339911, 93290, 1832001, 867207, 267181, 71688,
     1096668, 841912, 562154, 117103, 448891, 984707, 759005, 222805],
    [6714758, 2892014, 2200418, 814533, 5130625, 2392937, 927514, 373759,
     3171187, 1698142, 931572, 187962, 1039277, 1585175, 1284753, 371199],
    [589083, 195406, 100482, 30306, 524531, 276695, 150101, 39676, 474899,
     356308, 226229, 48683, 137124, 142619, 126619, 35417],
    [4283377, 1347796, 429873, 112833, 3616922, 1398501, 244454, 67006,
     2544528, 1010045, 407838, 81827, 651718, 357266, 237306, 60381],
    [3894142, 1306231, 653722, 207749, 2750932, 1368244, 570103, 165424,
     2521332, 1504167, 894712, 188475, 782946, 1003126, 865774, 248226],
    [1133560, 353461, 113047, 28214, 966642, 405831, 73024, 18446,
     1049707, 430777, 243521, 51239, 283139, 197143, 253712, 72444],
    [6197151, 2273374, 1352071, 462373, 4519188, 2309381, 805035, 265554,
     3719957, 2734197, 1642363, 345499, 1336249, 2362854, 2146357, 627783],
    [1066260, 333734, 102200, 25949, 799538, 352207, 50100, 12995,
     565256, 392333, 152749, 30507, 188706, 296864, 262010, 76396],
    [2227919, 706862, 252637, 68980, 1828584, 844389, 217351, 57723,
     1794434, 1004410, 416607, 85953, 490640, 375857, 261193, 69350],
    [589616, 190003, 100447, 24963, 544606, 441230, 187240, 43227,
     577335, 1075876, 775036, 164343, 470154, 1619556, 1609160, 484187],
    [8814885, 3169065, 1425281, 449155, 5889129, 2770893, 972365, 296251,
     3580582, 2323402, 1649231, 349223, 1088729, 1450712, 1270422, 359039],
    [292596, 91198, 35262, 9297, 262783, 130423, 39436, 10011, 241856,
     189463, 149361, 33175, 71186, 79486, 79578, 22413],
    [1239077, 396380, 226817, 65875, 1951492, 914073, 419534, 106546,
     1814504, 1269147, 972519, 211107, 611944, 942881, 888761, 258871],
    [996507, 315151, 95601, 24285, 649957, 284196, 42309, 11027, 482939,
     275841, 117698, 23293, 161112, 251086, 234114, 68953],
    [2601372, 814021, 266274, 64820, 2272617, 1001245, 198786, 47143,
     1868609, 1212664, 717633, 147013, 770098, 1650854, 2013948, 607009],
    [3772142, 1299167, 413709, 109563, 3152969, 1310780, 217685, 59124,
     2235739, 1174799, 658243, 136363, 673995, 812816, 928738, 272011],
    [4371411, 1350638, 387486, 94310, 4218321, 1549058, 141946, 36703,
     3415162, 1177943, 367251, 69897, 898595, 552350, 392083, 104714],
    [1585140, 500562, 169996, 42955, 1190515, 588917, 136751, 33535,
     1014630, 824980, 492080, 102399, 449073, 1044559, 973082, 291078],
    [318420, 163951, 257202, 99999, 270991, 206302, 268200, 84935, 247813,
     208613, 217729, 47507, 73370, 90529, 106431, 29735],
    [941560, 290731, 90583, 21851, 919390, 395600, 61620, 14520, 738843,
     492623, 217550, 43902, 269537, 484869, 458684, 135750],
    [4971239, 1555943, 499093, 128710, 4706616, 1779960, 305614, 81287,
     3075694, 1324978, 629877, 128849, 826644, 630183, 515322, 141588],
    [1346466, 404118, 150391, 33293, 1869452, 1040225, 208464, 45307,
     2153533, 2153251, 871128, 178600, 1022011, 2489279, 1620076, 475248],
    [1356463, 440998, 180116, 51469, 1051616, 482174, 193698, 52188,
     765994, 461350, 349375, 74871, 210009, 187410, 182836, 50697],
    [11955358, 3959869, 1491649, 441269, 8483852, 3196700, 586197,
     195512, 3622447, 1288634, 624642, 121320, 875024, 433315,
     315521, 77780],
    [1560431, 485512, 162864, 40702, 1531082, 673664, 131226, 33040,
     1453781, 854219, 463837, 98539, 407678, 364737, 386401, 110044],
    [1365000, 845170, 496793, 177816, 1059557, 675563, 375097, 121926,
     1156967, 715618, 476688, 100185, 347266, 403352, 299387, 82925],
    [999956, 317200, 106651, 26976, 673044, 356454, 84134, 20276, 564943,
     543198, 309354, 63336, 298073, 813213, 829694, 250182],
    [4298866, 1361857, 413289, 105651, 2892954, 1211145, 173599, 46369,
     1884902, 1005019, 467744, 93226, 630621, 998059, 1039301, 308175],
    [3391306, 1074772, 321507, 82923, 2406622, 903412, 112286, 32480,
     1160988, 442820, 247169, 49204, 292527, 190016, 245119, 69563],
    [5523049, 1758999, 520604, 134894, 3230182, 1269702, 151772, 45881,
     1424898, 575503, 267154, 50108, 417892, 529226, 583916, 171795],
    [3251978, 1015488, 314052, 78062, 2549059, 1160824, 172576, 42839,
     2135388, 1457146, 557769, 112120, 812245, 1545262, 946572, 275790],
    [9348444, 2992645, 955157, 258263, 5799233, 2281245, 450670, 130716,
     3037953, 961042, 512904, 102275, 735174, 324981, 252056, 62203],
    [456875, 139623, 73568, 17761, 661714, 389614, 146844, 33832,
     1067660, 806750, 608357, 130068, 408148, 767154, 1130679, 338447],
    [294002, 92838, 39614, 10030, 265022, 141356, 54775, 13614, 250670,
     224223, 213498, 46409, 89572, 163126, 347005, 104931],
    [3376526, 1076912, 323930, 85020, 1991091, 830540, 107159, 30884,
     849000, 557500, 229046, 44399, 330358, 703185, 429278, 124614],
    [2585452, 820391, 276323, 74846, 2180086, 818426, 184601, 50771,
     1208898, 452487, 247908, 50984, 302966, 162227, 129288, 33432],
    [2711104, 854426, 249944, 63436, 2067820, 766491, 77439, 21988,
     1055489, 422548, 157860, 29915, 295462, 293673, 253536, 72282],
    [4380356, 1401360, 426222, 111926, 2282337, 971336, 143421, 42570,
     1139179, 560531, 296281, 60158, 319459, 352877, 323722, 92425],
    [6042181, 1930908, 584717, 153245, 3256565, 1485803, 209824, 59900,
     2065465, 1310487, 489399, 94610, 764085, 1450610, 777191, 222567],
    [424424, 128858, 60276, 14082, 582396, 368170, 109284, 24577, 842083,
     830498, 459314, 97091, 383810, 901205, 803358, 238961],
    [1057336, 322439, 101474, 24785, 1178439, 453837, 71406, 17388,
     1122896, 450201, 247735, 52033, 300578, 201840, 255550, 72965]
                     ]
)

thompson_mean = np.mean(thompson, axis=0)
opt_mean = np.mean(opt, axis=0)
puct_mean = np.mean(puct, axis = 0)

thompson_mean = thompson_mean/(sum(thompson_mean))
opt_mean = opt_mean/sum(opt_mean)
puct_mean = puct_mean/sum(puct_mean)

lake_shape = (4, 4)
thompson_mean = thompson_mean.reshape(lake_shape)
opt_mean = opt_mean.reshape(lake_shape)
puct_mean = puct_mean.reshape(lake_shape)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 8),
                        subplot_kw={'xticks': [], 'yticks': []})

im0 = axs[0].imshow(opt_mean, interpolation='nearest', cmap='pink')
axs[0].set_title("W-MCTS-OS \n std=0.5 C=1.4142")
im1 = axs[1].imshow(thompson_mean, interpolation='nearest', cmap='pink')
axs[1].set_title("W-MCTS-TS \n std=0.5")
im2 = axs[2].imshow(puct_mean, interpolation='nearest', cmap='pink')
axs[2].set_title("Power-UCT \n C=1.4142")
cb0 = fig.colorbar(im0, ax=axs[0])
for t in cb0.ax.get_yticklabels():
    t.set_fontsize(20)
cb1 = fig.colorbar(im1, ax=axs[1])
for t in cb1.ax.get_yticklabels():
    t.set_fontsize(20)
cb1 = fig.colorbar(im2, ax=axs[2])
for t in cb1.ax.get_yticklabels():
    t.set_fontsize(20)
fig.suptitle('Relative visits on FrozenLake', fontsize=20)
plt.savefig('lake_visits.pdf')
plt.clf()


optWstein_visits = np.array([2.14568e+08, 2.34794e+08, 2.80489e+08, 2.12906e+08 ,1.43401e+08 ,1.01381e+08, 4.81705e+07, 3.58446e+08 ,3.0196e+08, 2.82e+08, 3.05768e+08, 1.90572e+08, 1.26293e+08, 5.90514e+07 ,3.34763e+08, 3.09723e+08, 2.94042e+08, 2.66169e+08, 2.00173e+08, 1.41951e+08, 6.61738e+07 ,3.83989e+08, 3.79769e+08, 4.33247e+08, 3.61139e+08, 2.35388e+08 ,1.72011e+08 ,8.98745e+07, 3.67017e+08, 4.28401e+08, 6.19759e+08 ,5.73164e+08 ,3.3136e+08 ,2.34691e+08, 9.59013e+07, 3.75489e+08, 5.41224e+08, 4.54874e+08, 3.88997e+08, 3.192e+08, 2.80819e+08, 1.07993e+08, 4.09174e+08, 7.62035e+08, 6.35846e+08, 5.78877e+08, 5.12586e+08 ,4.01015e+08, 2.97612e+08    ])
wstein_visits = np.array([ 1.0699e+09, 1.08503e+09, 1.72854e+09 ,8.76773e+08 ,7.28496e+08 ,6.81958e+08 ,1.45024e+09, 2.49687e+09, 1.17945e+09, 1.03631e+09 ,1.07132e+09 ,8.10247e+08, 7.7007e+08, 1.67178e+09, 9.78708e+08, 9.38865e+08 ,8.33447e+08 ,7.5825e+08 ,7.00957e+08, 6.93048e+08, 1.7366e+09 ,7.32525e+08, 7.60931e+08, 7.06643e+08 ,6.69075e+08, 6.4619e+08, 6.72264e+08, 2.16298e+09 ,5.86097e+08, 6.33742e+08, 6.91705e+08, 6.92264e+08, 6.2715e+08, 6.85458e+08 ,1.93347e+09, 5.01241e+08 ,5.54539e+08 ,5.50039e+08 ,5.56537e+08, 5.83899e+08, 8.59466e+08 ,2.04938e+09 ,4.06563e+08 ,5.30428e+08, 4.68299e+08, 4.7786e+08, 5.01531e+08, 5.7095e+08 ,1.67931e+09 ])
puct_visits2 = np.array([6.36227e+07, 6.82122e+07 ,7.30486e+07, 5.8728e+07, 4.12813e+07, 2.70068e+07, 1.35851e+07, 7.90816e+07, 7.54684e+07, 6.87973e+07, 6.61685e+07 ,4.67663e+07, 3.14593e+07, 1.75258e+07, 6.28579e+07, 6.00991e+07, 5.27855e+07, 4.51147e+07, 3.46079e+07, 2.3348e+07, 1.19053e+07, 4.97918e+07 ,4.73385e+07 ,4.0491e+07 ,3.37434e+07, 2.60613e+07, 1.76507e+07, 9.49002e+06, 3.61458e+07, 3.59936e+07 ,3.26617e+07 ,2.70868e+07 ,1.98457e+07, 1.35075e+07, 6.81648e+06, 2.80787e+07, 2.8616e+07, 2.49927e+07, 2.07837e+07, 1.60949e+07, 1.15243e+07, 5.51272e+06 ,2.15349e+07 ,2.36039e+07, 1.98189e+07, 1.65927e+07, 1.28357e+07 ,8.68577e+06 ,4.40606e+06  ])
puct_visits = np.array([8.17386e+08, 9.00922e+08, 8.98252e+08, 7.63426e+08, 6.14709e+08, 4.24403e+08 ,2.14903e+08, 1.05893e+09, 1.12367e+09 ,1.09205e+09 ,1.061e+09 ,7.77018e+08, 5.27291e+08, 2.63798e+08, 1.20439e+09 ,1.37672e+09, 1.39854e+09, 1.26276e+09 ,9.67879e+08, 6.35595e+08, 3.10343e+08 ,1.44501e+09, 1.74251e+09, 1.94871e+09 ,1.77492e+09, 1.25297e+09, 7.76445e+08 ,3.90073e+08 ,1.60213e+09, 2.04888e+09, 3.56729e+09, 3.3284e+09, 1.56171e+09 ,8.97188e+08, 4.16539e+08 ,1.58947e+09, 1.88602e+09, 2.11241e+09, 1.94738e+09, 1.39587e+09, 9.25346e+08, 4.20693e+08, 1.37778e+09, 1.65998e+09, 1.63041e+09 ,1.49259e+09, 1.15389e+09, 7.58319e+08, 3.73194e+08 ])



optWstein_visits = optWstein_visits/sum(optWstein_visits)
wstein_visits = wstein_visits/sum(wstein_visits)
puct_visits = puct_visits / sum(puct_visits)

arms_shape = (7,7)
optWstein_visits = optWstein_visits.reshape(arms_shape)
wstein_visits = wstein_visits.reshape(arms_shape)
puct_visits = puct_visits.reshape(arms_shape)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 8),
                        subplot_kw={'xticks': [], 'yticks': []})

im0 = axs[0].imshow(optWstein_visits, interpolation='nearest', cmap='pink',norm=matplotlib.colors.Normalize())
axs[0].set_title("W-MCTS-OS \n std=0.75 C=12.0")
im1 = axs[1].imshow(wstein_visits, interpolation='nearest', cmap='pink', norm=matplotlib.colors.Normalize())
axs[1].set_title("W-MCTS-TS \n std=0.75")
im2 = axs[2].imshow(puct_visits, interpolation='nearest', cmap='pink', norm=matplotlib.colors.Normalize())
axs[2].set_title("D2NG")
cb0 = fig.colorbar(im0, ax=axs[0])
for t in cb0.ax.get_yticklabels():
    t.set_fontsize(20)
cb1 = fig.colorbar(im1, ax=axs[1])
for t in cb1.ax.get_yticklabels():
    t.set_fontsize(20)
cb1 = fig.colorbar(im2, ax=axs[2])
for t in cb1.ax.get_yticklabels():
    t.set_fontsize(20)

fig.suptitle('Relative visits on Rocksample [7x8]', fontsize=20)
plt.savefig('rocksample_visits.pdf')

dvg = 0