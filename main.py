import random
from run_modele import model_to_test
import torch
from tqdm import tqdm
import json
from pathlib import Path
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

search_param = {
    "nb_epoch": 2000,  # epoch number
    "save_rate": 50,  # rate to save
    "weight_data": 1,
    "weight_pde": 1,
    "Re": 100,
    "gamma_scheduler": 0.999,  # Gamma scheduler for lr
    "nb_layers": 1,
    "N_pde":1000000,
    "batch_size_pde":6000,
    "nb_neurons": 32,
    "n_pde_test": 5000,
    "n_data_test": 5000,
    "lr_init": 1e-3,
    "x_min": 0.15,
    "x_max": 0.325,
    "y_min": -0.1,
    "y_max": 0.1,
    "t_min": 4,
    "t_max": 6,
    "transfert_learning": "None",
}

save_folder = '1_first_try'
time_run = 18
nb_test = 3
batch_size_min = 1000
batch_size_max = 10000
N_tot_min = 10000
N_tot_max = 80000



int_parameter = {
    "batch_size_min":batch_size_min,
    "batch_size_max":batch_size_max,
    "N_tot_min":N_tot_min,
    "N_tot_max":N_tot_max,
    "time_run":time_run,
    'nb_test':nb_test
}

Path('results/' + save_folder).mkdir(parents=True, exist_ok=True)  # Creation du dossier de result
with open('results/' + save_folder + "/param_simu.json", "w") as file:
    json.dump(search_param, file, indent=4)
    json.dump(int_parameter, file, indent=4)



for num_sim in tqdm(range(nb_test)):
    print('\n\n\n-------------------------')
    print(f"Simu n°{num_sim+1}")
    print('-------------------------\n')
    batch_size = int(random.uniform(batch_size_min, batch_size_max))
    N_tot = int(random.uniform(N_tot_min, N_tot_max))#int(np.exp(np.log(10)*np.random.uniform(np.log10(N_tot_min), np.log10(N_tot_max))))
    to_test = model_to_test(
        batch_size,
        N_tot,
        save_folder,
        time_run,
        num_sim,
        device,
        search_param
    )
    to_test.run()
    
    