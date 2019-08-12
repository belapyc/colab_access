import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_prep import DataPrep
from __init__ import PARAMETERS, FILE_ADDRESS
# , "APPL" "RL_1":"RL","PM-1":"PM-1","NVID":"Nvidia",  "NQ10":"NQ10",
instruments= { "NFLX":"NFLX",  "APPL":"Apple","eure":"eure", "RL_1":"RL","PM-1":"PM-1","NVID":"Nvidia", "BAC_":"BAC_"}
random_seeds = [0,1,5]
forests_no = [5,10,20]

for instrument in instruments:
    print(instrument)
    print(instruments[instrument])
    for year in PARAMETERS['ALL_YEARS']:
        for forest_no in forests_no:
            for rnseed in random_seeds:
                file_exists = True
                try:
                    preds_forest = pd.read_csv("predictions/forest/rn"+str(rnseed)+"f"+str(forest_no)+"/"+instrument+"_forest_True_wavelet_True_randomSeed_"+str(rnseed)+"_forest_no_"+str(forest_no)+"_YEAR_"+str(year)+"predictions")
                    preds_sae = pd.read_csv("predictions/SAE/rn"+str(rnseed)+"/"+instrument+"_forest_False_wavelet_True_randomSeed_"+str(rnseed)+"_forest_no_10_YEAR_"+str(year)+"predictions")
                except OSError as e:
                    file_exists = False
                if file_exists:
                    actual = pd.read_csv("actual_values/"+instrument+"_YEAR_"+str(year)+"actual.csv")
                    with plt.style.context('dark_background'):
                        plt.figure()
                        plt.plot(preds_forest.to_numpy()[:,0],preds_forest.to_numpy()[:,1],'b--', label="predictions forest", linewidth=1)
                        plt.plot(preds_sae.to_numpy()[:,0],preds_sae.to_numpy()[:,1],'r--', label="predictions sae", linewidth=1)
                        plt.xlabel("Tick")
                        plt.ylabel("Price")
                        plt.plot( actual.to_numpy()[:,0],actual.to_numpy()[:,1], label="actual", linewidth=1)
                        plt.title("Performance of year "+str(year))
                        plt.legend()
                    save_plot_address = "Plots/both/rn"+str(rnseed)+"_f"+str(forest_no)+"/"+instrument+"_randomSeed_"+str(rnseed)+""+"_forest_no_"+str(forest_no)+"_YEAR_"+str(year)+".pdf"
                    plt.savefig(save_plot_address, bbox_inches='tight', dpi = 150)
                    plt.close()
        # plt.show()

# year = 2009
# preds_forest = pd.read_csv("predictions/forest/"+"RL_1"+"_forest_True_wavelet_True_randomSeed_0_forest_no_10_YEAR_"+str(year)+"predictions")
# preds_sae = pd.read_csv("predictions/SAE/rn0/"+"RL_1"+"_forest_False_wavelet_True_randomSeed_0_forest_no_10_YEAR_"+str(year)+"predictions")
# actual = pd.read_csv("actual_values/"+"RL_1"+"_YEAR_"+str(year)+"actual.csv")
#
# with plt.style.context('dark_background'):
#     plt.figure()
#     plt.plot(preds_forest.to_numpy()[:,0],preds_forest.to_numpy()[:,1],'b--', label="predictions forest", linewidth=1)
#     plt.plot(preds_sae.to_numpy()[:,0],preds_sae.to_numpy()[:,1],'r--', label="predictions sae", linewidth=1)
#     plt.xlabel("Tick")
#     plt.ylabel("Price")
#     plt.plot( actual.to_numpy()[:,0],actual.to_numpy()[:,1], label="actual", linewidth=1)
#     plt.title("Performance of year 2009")
#     plt.legend()
#     save_plot_address = "Plots/both/"+"RL_1"+"_randomSeed_0"+"_forest_no_10_YEAR_"+str(year)+".pdf"
#     plt.savefig(save_plot_address, bbox_inches='tight', dpi = 150)

# print(preds_forest.shape)
# print(preds_sae.shape)
# print(actual.shape)
# print(preds_forest.head(5))
print(preds_forest.to_numpy()[:,1])


#
# with plt.style.context('dark_background'):
#     plt.figure()
#     plt.plot(preds_forest.to_numpy()[:,0],preds_forest.to_numpy()[:,1],'b--', label="predictions forest", linewidth=1)
#     plt.plot(preds_sae.to_numpy()[:,0],preds_sae.to_numpy()[:,1],'r--', label="predictions sae", linewidth=1)
#     plt.xlabel("Tick")
#     plt.ylabel("Price")
#     plt.plot( actual.to_numpy()[:,0],actual.to_numpy()[:,1], label="actual", linewidth=1)
#     plt.title("Performance of year 2009")
#     plt.legend()
# plt.show()
#
# fig = plt.figure()
# ax = plt.subplot(111)
# ax.plot(preds_forest.to_numpy()[:,0],preds_forest.to_numpy()[:,1], label='Forest predictions')
# ax.plot(preds_sae.to_numpy()[:,0],preds_sae.to_numpy()[:,1], label='SAE predictions')
# ax.plot(actual.to_numpy()[:,0],actual.to_numpy()[:,1], label='actual values')
# plt.title('Legend outside')
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
# ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
# plt.show()
