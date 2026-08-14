# numactl --interleave=all python3 data_bundle_creation_nmf.py # Is not used anymore because of ProcessPoolExecutor 
# python3 data_bundle_creation_nmf.py

import os

os.environ["KMP_BLOCKTIME"] = "0"

PHYSICAL_CORES = 56
N_THREADS = 112

os.environ["POLARS_MAX_THREADS"] = str(N_THREADS)

import sys
#!pip install polars --target ./my_custom_packages
#sys.path.append("./my_custom_packages") #
custom_path = "/stor/home/he4249/orange/my_custom_packages" 
sys.path.insert(0, custom_path)



import polars as pl
import torch
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import time
import numpy as np

import concurrent.futures

import multiprocessing as mp

pl.Config.set_tbl_rows(50)
pl.Config(tbl_cols = 50)


print("Setting up neural network process.")

class latent_nmf(nn.Module):
    def __init__(self, n_users, n_plans, k, global_mean):
        # Parent class
        super().__init__()

        self.global_mean = global_mean

        if k > 0:
            # Sparse was not adpated

            self.user_embedding = nn.Embedding(n_users, k) # max_norm = 3.0 destroyed speed and cpu utilisation, clamped after every batch once
            self.plan_embedding = nn.Embedding(n_plans, k)

            nn.init.uniform_(self.user_embedding.weight, a = -0.5, b = 0.5)
            nn.init.uniform_(self.plan_embedding.weight, a = -0.5, b = 0.5)

        else:
            self.user_embedding = None
            self.plan_embedding = None

        self.user_bias = nn.Embedding(n_users, 1)
        self.plan_bias = nn.Embedding(n_plans, 1)

        nn.init.uniform_(self.user_bias.weight, a = 0.01, b = 0.2)
        nn.init.uniform_(self.plan_bias.weight, a = 0.01, b = 0.2)


        # Small positive range to control loss on start of otpimisation
        # Weights mapped by softplus give approx [0.01, 0.2]
        #nn.init.uniform_(self.user_embedding.weight, a = -4.5, b = -1.5)
        #nn.init.uniform_(self.plan_embedding.weight, a = -4.5, b = -1.5)




    def forward(self, user_i, plan_i, k):

        # user_k = torch.relu(self.user_embedding(user_i))
        # plan_k = torch.relu(self.plan_embedding(plan_i))

        if k > 0:
            user_k = torch.nn.functional.softplus(self.user_embedding(user_i))
            plan_k = torch.nn.functional.softplus(self.plan_embedding(plan_i))
            interaction = (user_k * plan_k).sum(dim = 1) / (k ** 0.5)
        else:
            interaction = 0.0

        user_b = self.user_bias(user_i).squeeze()
        plan_b = self.plan_bias(plan_i).squeeze()
                

        uptca_prediction = interaction + user_b + plan_b + self.global_mean

        return uptca_prediction



def train_k_model(k, train_data, test_data, n_users, n_plans, n_cores, batch_size, min_change, limit, lr, max_epochs, GLOBAL_MEAN, worker_idx, weight_decay, n_neg, neg_weight, epoch_testing):

    '''
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    os.environ["OMP_NUM_THREADS"] = str(n_cores)
    torch.set_num_threads(n_cores)
    '''

    ''''
    # Intel Math Kernel Library for pytorch to force a limit
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    # Open Basic Linear Algebra Subprograms for NumPy and SciPy to force a limit
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
    # Open Multi Processing for faiss
    os.environ["OMP_NUM_THREADS"] = str(n_cores)

    os.environ["POLARS_MAX_THREADS"] = str(N_THREADS)

    torch.set_num_threads(n_cores)

    '''

    time_run = time.time()

    # Socket 0 CPUs all even IDs, 56 
    node0_cpus = [i for i in range(N_THREADS) if i % 2 == 0][:n_cores]
    # Socket 1 CPUs all odd IDs, 56 
    node1_cpus = [i for i in range(N_THREADS) if i % 2 != 0][:n_cores]


    # Even worker_idx -> Socket 0, Odd worker_idx -> Socket 1
    socket_id = worker_idx % 2
    my_cpus = node0_cpus if socket_id == 0 else node1_cpus

    # Lock process to entire socket: 28 physical + 28 threads
    os.sched_setaffinity(0, set(my_cpus)) # Doens't work or only worked for main process
    
    os.environ["OMP_NUM_THREADS"] = str(n_cores)
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    torch.set_num_threads(n_cores)

    train_user = train_data[0].clone()
    train_plan = train_data[1].clone()
    train_uptca = train_data[2].clone()

    test_user = test_data[0].clone()
    test_plan = test_data[1].clone()
    test_uptca = test_data[2].clone()

    train_size = len(train_user)
    test_size = len(test_user)

    curr_seed = torch.Generator().manual_seed(4 + k)

    
    opt_model = latent_nmf(n_users = n_users, n_plans = n_plans, k = k, global_mean = GLOBAL_MEAN)
    loss_function = nn.MSELoss()

    # Adam because of major popular plans in data while simultaneously having very rare plans, AdamW for weight decay
    #optimiser = optim.AdamW(opt_model.parameters(), lr = lr, weight_decay = 0.01)
    '''
    user_params = [opt_model.user_bias.weight]
    plan_params = [opt_model.plan_bias.weight]

    if k > 0:
        user_params.append(opt_model.user_embedding.weight)
        plan_params.append(opt_model.plan_embedding.weight)
    '''

    if k == 0:
        optimiser = optim.AdamW([
            {"params": [opt_model.user_bias.weight], "weight_decay": weight_decay, "lr": lr},
            {"params": [opt_model.plan_bias.weight], "weight_decay": weight_decay * 100, "lr": lr} # might need to remove the multiplier to weight decay.
        ])
    else:
        optimiser = optim.AdamW([
            {"params": [opt_model.user_bias.weight], "weight_decay": weight_decay, "lr": lr},
            {"params": [opt_model.plan_bias.weight], "weight_decay": weight_decay * 100, "lr": lr},
            {"params": [opt_model.user_embedding.weight], "weight_decay": weight_decay, "lr": lr},
            {"params": [opt_model.plan_embedding.weight], "weight_decay": weight_decay, "lr": lr},
        ])

    scheduler = ReduceLROnPlateau(
            optimiser, 
            mode = "min",
            factor = 0.5,
            patience = 4,
            threshold = min_change * 10,
            threshold_mode = "rel",
            #verbose = True
    )

    #prev_loss = float("inf")
    stop_limit = 0
    best_loss = float("inf")
    

    opt_model.train()

    for epoch in range(1, max_epochs + 1):
        
        t0 = time.time()
        
        batch_perm = torch.randperm(train_size, generator = curr_seed)
        bp_user = train_user[batch_perm]
        bp_plan = train_plan[batch_perm]
        bp_uptca = train_uptca[batch_perm]

        # Going back to per step instead of per batch, noticed that with this change k = 0 (only biases) has the best loss score,
        #neg_bg_plan = torch.randint(0, n_plans, (train_size * n_neg, ))
        #neg_bg_target = torch.zeros(train_size * n_neg, dtype = torch.float32)

        train_loss = 0
        for i in range(0, train_size, batch_size):
            
            user_i_batch = bp_user[i : i + batch_size]
            plan_i_batch = bp_plan[i : i + batch_size]
            ca_v_batch = bp_uptca[i : i + batch_size]
            
            N_curr = len(user_i_batch)

            # Going back to per step
            '''
            neg_user_batch = user_i_batch.repeat_interleave(n_neg)
            neg_plan_batch = neg_bg_plan[i * n_neg : (i + N_curr) * n_neg]
            neg_target_batch = neg_bg_target[i * n_neg : (i + N_curr) * n_neg]
            '''

            # Treating every user the same at random, could use word2vec-style negative sampling
            neg_user_batch = user_i_batch.repeat_interleave(n_neg)
            neg_plan_batch = torch.randint(0, n_plans, (N_curr * n_neg,))
            neg_target_batch = torch.zeros(N_curr * n_neg, dtype=torch.float32)

            optimiser.zero_grad()

            pos_predictions = opt_model(user_i_batch, plan_i_batch, k)
            pos_loss = loss_function(pos_predictions, ca_v_batch)

            neg_predictions = opt_model(neg_user_batch, neg_plan_batch, k)
            neg_loss = loss_function(neg_predictions, neg_target_batch)

            #loss = pos_loss + neg_loss
            loss = pos_loss + neg_weight * neg_loss

            loss.backward()

            train_loss += loss.item() * N_curr * (n_neg + 1)

            # Bounding the gradiants so no singular batch runs everything
            # Wasn't engaged even once during the runs
            last_grad_norm = torch.nn.utils.clip_grad_norm_(opt_model.parameters(), max_norm = 1.0)

            optimiser.step()
    
        if k > 0:
            with torch.no_grad():
                # Euclidean distance
                # data.renorm_  
                opt_model.user_embedding.weight.renorm_(p=2, dim=0, maxnorm=3.0)
                opt_model.plan_embedding.weight.renorm_(p=2, dim=0, maxnorm=3.0)

        if epoch % epoch_testing == 0:
            opt_model.eval()
            opt_loss = 0.0

            # Maybe should be outside of the testing loop.
            with torch.no_grad():
                for i in range(0, test_size, batch_size):
                    user_i_batch = test_user[i : i + batch_size]
                    plan_i_batch = test_plan[i : i + batch_size]
                    ca_v_batch = test_uptca[i : i + batch_size]

                    predictions = opt_model(user_i_batch, plan_i_batch, k)
                    loss = loss_function(predictions, ca_v_batch)
                    opt_loss += loss.item() * len(user_i_batch)

            #d_loss = (prev_loss - opt_loss)/prev_loss
            #prev_loss = opt_loss

            if opt_loss < best_loss * (1.0 - min_change):
                best_loss = opt_loss
                stop_limit = 0
            else:
                stop_limit += 1

            scheduler.step(opt_loss)
            
            opt_model.train()

            if stop_limit >= limit:
                print(f"k = {k}, converged at epoch {epoch - (epoch_testing * limit) + epoch_testing }, last epoch is {epoch}, and best loss of the run is {best_loss:.2f}")
                break

            if k > 0:
                sp_user = torch.nn.functional.softplus(opt_model.user_embedding.weight)
                sp_plan = torch.nn.functional.softplus(opt_model.plan_embedding.weight)
                user_emb_mean = sp_user.mean().item()
                user_emb_max = sp_user.max().item()
                plan_emb_mean = sp_plan.mean().item()
                plan_emb_max = sp_plan.max().item()
            else:
                user_emb_mean, user_emb_max, plan_emb_mean, plan_emb_max = 0.0, 0.0, 0.0, 0.0
            
            if epoch % epoch_testing == 0:
                print(f"{epoch_testing} epoch took {time.time() - t0:.2f}s")


            user_bias_mean = opt_model.user_bias.weight.mean().item()
            plan_bias_mean = opt_model.plan_bias.weight.mean().item()

            #total_norm = torch.nn.utils.clip_grad_norm_(opt_model.parameters(), max_norm = 1.0)
            d_loss = (opt_loss - best_loss) / best_loss * 100


            current_lr = scheduler.get_last_lr()[0] 
            print(f"For k = {k} running on epoch {epoch}, with loss {opt_loss} and lr {current_lr}, and best loss of {best_loss}")
            print(f"user_emb_mean {user_emb_mean} and user_emb_max {user_emb_max} and plan_emb_mean {plan_emb_mean} and plan_emb_max {plan_emb_max}")
            print(f"user_bias_mean {user_bias_mean} and plan_bias_mean {plan_bias_mean} and last_grad_norm {last_grad_norm} and d_loss {d_loss}")
            print(f"current train loss {train_loss}")
            print(f"curren stop limit {stop_limit}")
        
            

    print(f"Run took {time.time() - time_run}s")
    return {"k": k, "best_loss": best_loss}





if __name__ == "__main__":

    c = mp.get_context("spawn")

    # Creating tensor data

    print("Preparing data.")

    # CA by plan by user
    df_merged = pl.read_parquet("df_merged.parquet")

    df_user_plan_tca = df_merged.group_by(["msisdn", "Nom du forfait"]).agg(
        pl.col("CA").sum().alias("UPTCA")
    )

    # No better results
    '''
    df_user_plan_tca = (
    df_merged
    .filter(pl.len().over("msisdn") >= 3)
    .group_by(["msisdn", "Nom du forfait"])
    .agg(pl.col("CA").sum().alias("UPTCA"))
    )
    '''

    unique_users = df_user_plan_tca.select("msisdn").unique().with_row_index("user_i")
    unique_plans = df_user_plan_tca.select("Nom du forfait").unique().with_row_index("plan_i")

    df_uptca_nzv = df_user_plan_tca.join(unique_users, on = "msisdn").join(unique_plans, on = "Nom du forfait")

    n_users = unique_users.shape[0]
    n_plans = unique_plans.shape[0]
    plan_names = unique_plans["Nom du forfait"].unique().to_list()

    # 1d tensors
    user_indices = torch.tensor(df_uptca_nzv["user_i"].to_numpy(), dtype = torch.long) # .share_memory_()
    plan_indices = torch.tensor(df_uptca_nzv["plan_i"].to_numpy(), dtype = torch.long) # .share_memory_()
    # Logged revenue
    ca_values = torch.tensor(np.log1p(df_uptca_nzv["UPTCA"].to_numpy()), dtype = torch.float32) # .share_memory_()

    GLOBAL_MEAN = ca_values.mean().item()

    n_total = len(user_indices)
    seed = torch.Generator().manual_seed(4)



    #tensordata = TensorDataset(user_indices, plan_indices, ca_values)
    ## Too slow, only loads in one memory but we have interleave nodes here
    #data_batched = DataLoader(tensordata, batch_size = 16384, shuffle = True)



    opt = True
    # Use python3
    if opt:
        print("Starting parameter optimisation")

        time_opt = time.time()
        
        test_size = int(0.20 * n_total)
        train_size = n_total - test_size

        perm = torch.randperm(n_total, generator = seed)

        train_idx = perm[:train_size]
        test_idx = perm[train_size:]

        train_user = user_indices[train_idx] # .share_memory_()
        train_plan = plan_indices[train_idx] # .share_memory_()
        train_uptca = ca_values[train_idx] # .share_memory_()

        test_user = user_indices[test_idx] # .share_memory_()
        test_plan = plan_indices[test_idx] # .share_memory_()
        test_uptca = ca_values[test_idx] # .share_memory_()

        train_data = (train_user, train_plan, train_uptca)
        test_data = (test_user, test_plan, test_uptca)

        k_candidates = [0, 1, 2, 3, 4, 6, 8, 12, 16]
        n_workers = 2
        n_cores = 56  # 56 threads per worker
        max_epochs = 500
        batch_size = 131072 
        min_change = 0.0001
        limit = 8
        lr = 0.001
        weight_decay = 0.01
        n_neg = 4
        neg_weight = 1
        epoch_testing = 2

        scores = []

        with concurrent.futures.ProcessPoolExecutor(max_workers = n_workers, mp_context = c) as executor:
            futures = []
            for worker_idx, k in enumerate(k_candidates):
                futures.append(
                    executor.submit(
                            train_k_model, k, train_data, test_data, 
                            n_users, n_plans, n_cores, batch_size, min_change, limit, lr, max_epochs, GLOBAL_MEAN, worker_idx, weight_decay, n_neg, neg_weight, epoch_testing
                        ) 
                )
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    scores.append(result)
                except Exception as e:
                    print(f"Error {e}")
                    import traceback
                    traceback.print_exc()
                

        df_scores = pl.DataFrame(scores).sort("best_loss")
        print(df_scores)

        print()
        print(f"Time taken {time.time() - time_opt:.2f}s")














# Conclusion: data innapropriate for nn nmf model.



### LOGS



# Logs, hidden
'''


    ┌─────┬───────────┐
    │ k   ┆ best_loss │
    │ --- ┆ ---       │
    │ i64 ┆ f64       │
    ╞═════╪═══════════╡
    │ 4   ┆ 1.4581e6  │
    │ 3   ┆ 1.4607e6  │
    │ 8   ┆ 1.4795e6  │
    │ 16  ┆ 1.4992e6  │
    │ 32  ┆ 1.5254e6  │
    │ 64  ┆ 1.5373e6  │
    └─────┴───────────┘

    Time taken 3072.59s # pod sharing constraints
    
    all K converged at epoch 6 with their last epoch being 39

    Keeping k = 4??



For k = 4 running on epoch 30, with loss 3646117.326528549 and lr 0.01
k = 4, converged at epoch 6, last epoch is 39, and best loss of the run is 1464814.61
Run took 267.7710630893707s
For k = 8 running on epoch 30, with loss 3064203.7067813873 and lr 0.01
k = 8, converged at epoch 6, last epoch is 39, and best loss of the run is 1475812.39
Run took 345.0376446247101s
For k = 3 running on epoch 30, with loss 3500386.8515529633 and lr 0.01
k = 3, converged at epoch 6, last epoch is 39, and best loss of the run is 1453756.79
Run took 647.7444677352905s
For k = 16 running on epoch 30, with loss 2370461.8732643127 and lr 0.01
k = 16, converged at epoch 6, last epoch is 39, and best loss of the run is 1494922.80
Run took 430.9316084384918s
For k = 32 running on epoch 30, with loss 2023742.658296585 and lr 0.01
k = 32, converged at epoch 6, last epoch is 39, and best loss of the run is 1528344.50
Run took 826.2606210708618s
For k = 64 running on epoch 30, with loss 1817101.8979635239 and lr 0.01
k = 64, converged at epoch 6, last epoch is 39, and best loss of the run is 1553141.13
Run took 1173.7502255439758s
shape: (6, 2)
┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 3   ┆ 1.4538e6  │
│ 4   ┆ 1.4648e6  │
│ 8   ┆ 1.4758e6  │
│ 16  ┆ 1.4949e6  │
│ 32  ┆ 1.5283e6  │
│ 64  ┆ 1.5531e6  │
└─────┴───────────┘

Time taken 2220.16s



Adding weight decay (AdamW), lr reduced to 0.005 from 0.02, added layers to separate embeddings reduction given size disparity, increased the starting weight for softplus, adding a scaling factor to predictions to compare K on similar grounds



k = 0 has a better loss score. Biases are dictating the data.
k = 0, converged at epoch 9, last epoch is 42, and best loss of the run is 1416377.27

all k converge at epoch 9 with the changes made


adding embeddings never beats the bias model on held out data
the embeddings are negative for generalisation
every extra parameter they add is being used to memorize training noise


using only main mobile plans reduced loss but k = 0 is still better
using only users with a minimum of three transactions reduced loss but k = 0 us still better

Increasing batch size from 65536 to 131072
k = 0, converged at epoch 15, last epoch is 48, and best loss of the run is 1292304.96
k = 3, converged at epoch 12, last epoch is 45, and best loss of the run is 1354404.68

x
With n_neg of 4 (added feature)
┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 3   ┆ 2.9357e6  │
│ 0   ┆ 3.7583e6  │
└─────┴───────────┘

added neg_weight, max norm

batch size to 262144
┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 4   ┆ 2.3280e6  │
│ 16  ┆ 2.6089e6  │
└─────┴───────────┘

k = 4: 10s per 3 epoch
k = 16: 14s per 3 epoch

batch size to 32768
┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 4   ┆ 2.3280e6  │
│ 16  ┆ 2.6089e6  │
└─────┴───────────┘

k = 4: 20s per 3 epoch
k = 16: 14s per 3 epoch


keeping 262144 for batch size for speed until perhaps readjustments on testing later on.

┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 0   ┆ 2.2310e6  │
│ 3   ┆ 2.2830e6  │
└─────┴───────────┘


increasing neg_weight to 1 from 0.2(testing purposes)

┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 0   ┆ 2.5621e6  │
│ 3   ┆ 2.6410e6  │
└─────┴───────────┘

bringing back neg_weight to 0.2 but with max_norm = None (testing purposes))

┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 0   ┆ 2.2331e6  │
│ 3   ┆ 2.3042e6  │
└─────┴───────────┘

Went back to per step randint negs
using neg_weight = 1 and max_norm = 3

┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 0   ┆ 2.5654e6  │
│ 3   ┆ 2.6503e6  │
└─────┴───────────┘

Decreased batch size to 131072 down from 262144

┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 3   ┆ 2.6207e6  │
│ 0   ┆ 2.7575e6  │
└─────┴───────────┘

NEVER CHANGING BATCH SIZE AGAIN: batch_size = 131072

┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 4   ┆ 2.5610e6  │
│ 16  ┆ 2.8464e6  │
└─────┴───────────┘

lr down to 0.002 from 0.005, 



'''
# Final tests
'''
initial lr = 0.002, testing every epoch

┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 0   ┆ 2.4919e6  │
│ 2   ┆ 2.5549e6  │
│ 1   ┆ 2.5555e6  │
│ 4   ┆ 2.5865e6  │
│ 8   ┆ 2.5912e6  │
│ 3   ┆ 2.5991e6  │
│ 6   ┆ 2.6338e6  │
│ 12  ┆ 2.7056e6  │
│ 16  ┆ 2.7989e6  │
└─────┴───────────┘
Time taken 1062.29s

initial lr = 0.01, testing every epoch

┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 1   ┆ 2.5224e6  │
│ 0   ┆ 2.6091e6  │
│ 3   ┆ 2.6159e6  │
│ 8   ┆ 2.6488e6  │
│ 2   ┆ 2.6544e6  │
│ 6   ┆ 2.6613e6  │
│ 4   ┆ 2.6804e6  │
│ 12  ┆ 2.7104e6  │
│ 16  ┆ 2.8450e6  │
└─────┴───────────┘
Time taken 602.34s

initial lr = 0.005, testing every epoch

┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 1   ┆ 2.5128e6  │
│ 0   ┆ 2.5213e6  │
│ 2   ┆ 2.5940e6  │
│ 4   ┆ 2.5942e6  │
│ 3   ┆ 2.6012e6  │
│ 8   ┆ 2.6362e6  │
│ 6   ┆ 2.6524e6  │
│ 12  ┆ 2.6999e6  │
│ 16  ┆ 2.8081e6  │
└─────┴───────────┘

Time taken 749.12s



changing epoch testing from 1 to 5 (limit was thus reduced to 4 down from 8)(when lr was 0.005 and epoch testing was 3 it gave better results)

┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 12  ┆ 2.7240e6  │
│ 8   ┆ 2.7437e6  │
│ 6   ┆ 2.7844e6  │
│ 4   ┆ 2.8169e6  │
│ 16  ┆ 2.8371e6  │
│ 3   ┆ 2.8567e6  │
│ 2   ┆ 2.8932e6  │
│ 1   ┆ 2.9426e6  │
│ 0   ┆ 3.0913e6  │
└─────┴───────────┘

Time taken 1503.76s

initial lr = 0.001, testing every 2 epoch (limit was brought back to 8)

┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 0   ┆ 2.4841e6  │
│ 1   ┆ 2.5544e6  │
│ 2   ┆ 2.5897e6  │
│ 3   ┆ 2.5934e6  │
│ 4   ┆ 2.5961e6  │
│ 6   ┆ 2.6522e6  │
│ 8   ┆ 2.6650e6  │
│ 12  ┆ 2.6750e6  │
│ 16  ┆ 2.7940e6  │
└─────┴───────────┘

'''
# Last run logs
'''
he4249@lambcomp02:~/orange$ python3 data_bundle_creation.py
Setting up neural network process.
Preparing data.
Starting parameter optimisation
Setting up neural network process.
Setting up neural network process.
2 epoch took 5.56s
For k = 0 running on epoch 2, with loss 2523188.894180298 and lr 0.001, and best loss of 2523188.894180298
user_emb_mean 0.0 and user_emb_max 0.0 and plan_emb_mean 0.0 and plan_emb_max 0.0
user_bias_mean 0.08469251543283463 and plan_bias_mean -0.02078906260430813 and last_grad_norm 1.373365879058838 and d_loss 0.0
current train loss 2249743499.244385
curren stop limit 0
2 epoch took 10.31s
For k = 1 running on epoch 2, with loss 3192004.5766353607 and lr 0.001, and best loss of 3192004.5766353607
user_emb_mean 0.6934627294540405 and user_emb_max 0.9737008810043335 and plan_emb_mean 0.6369962692260742 and plan_emb_max 0.8983818292617798
user_bias_mean 0.08463830500841141 and plan_bias_mean -0.017283760011196136 and last_grad_norm 1.6623444557189941 and d_loss 0.0
current train loss 2539662559.4067383
curren stop limit 0
2 epoch took 5.31s
For k = 0 running on epoch 4, with loss 2484066.200273514 and lr 0.001, and best loss of 2484066.200273514
user_emb_mean 0.0 and user_emb_max 0.0 and plan_emb_mean 0.0 and plan_emb_max 0.0
user_bias_mean 0.06265635043382645 and plan_bias_mean -0.13065409660339355 and last_grad_norm 1.3202126026153564 and d_loss 0.0
current train loss 2166367417.5195312
curren stop limit 0
2 epoch took 6.93s
For k = 0 running on epoch 6, with loss 2492543.598022461 and lr 0.001, and best loss of 2484066.200273514
user_emb_mean 0.0 and user_emb_max 0.0 and plan_emb_mean 0.0 and plan_emb_max 0.0
user_bias_mean 0.04050346836447716 and plan_bias_mean -0.22799520194530487 and last_grad_norm 1.283925175666809 and d_loss 0.3412710075123489
current train loss 2092870459.2712402
curren stop limit 1
2 epoch took 6.97s
For k = 0 running on epoch 8, with loss 2528371.9264526367 and lr 0.001, and best loss of 2484066.200273514
user_emb_mean 0.0 and user_emb_max 0.0 and plan_emb_mean 0.0 and plan_emb_max 0.0
user_bias_mean 0.018366865813732147 and plan_bias_mean -0.3139829933643341 and last_grad_norm 1.2406729459762573 and d_loss 1.783596837082866
current train loss 2027872197.557373
curren stop limit 2
2 epoch took 17.43s
For k = 1 running on epoch 4, with loss 2881852.2494716644 and lr 0.001, and best loss of 2881852.2494716644
user_emb_mean 0.6832738518714905 and user_emb_max 0.9733270406723022 and plan_emb_mean 0.5847899317741394 and plan_emb_max 0.830136775970459
user_bias_mean 0.06237134709954262 and plan_bias_mean -0.1302243322134018 and last_grad_norm 1.5593891143798828 and d_loss 0.0
current train loss 2415719932.3669434
curren stop limit 0
2 epoch took 7.36s
For k = 0 running on epoch 10, with loss 2580451.1572589874 and lr 0.001, and best loss of 2484066.200273514
user_emb_mean 0.0 and user_emb_max 0.0 and plan_emb_mean 0.0 and plan_emb_max 0.0
user_bias_mean -0.0037144492380321026 and plan_bias_mean -0.3898278772830963 and last_grad_norm 1.2109884023666382 and d_loss 3.880128354665465
current train loss 1970131981.3586426
curren stop limit 3
2 epoch took 5.10s
For k = 0 running on epoch 12, with loss 2641658.8101882935 and lr 0.001, and best loss of 2484066.200273514
user_emb_mean 0.0 and user_emb_max 0.0 and plan_emb_mean 0.0 and plan_emb_max 0.0
user_bias_mean -0.025722797960042953 and plan_bias_mean -0.45662763714790344 and last_grad_norm 1.187395453453064 and d_loss 6.344138892008094
current train loss 1918644142.232666
curren stop limit 4
2 epoch took 9.83s
For k = 1 running on epoch 6, with loss 2696087.85515213 and lr 0.001, and best loss of 2696087.85515213
user_emb_mean 0.6735654473304749 and user_emb_max 0.9729543924331665 and plan_emb_mean 0.5373920202255249 and plan_emb_max 0.7665863037109375
user_bias_mean 0.03985854983329773 and plan_bias_mean -0.23109085857868195 and last_grad_norm 1.4828826189041138 and d_loss 0.0
current train loss 2307578634.708252
curren stop limit 0
2 epoch took 5.13s
For k = 0 running on epoch 14, with loss 2709015.296705246 and lr 0.0005, and best loss of 2484066.200273514
user_emb_mean 0.0 and user_emb_max 0.0 and plan_emb_mean 0.0 and plan_emb_max 0.0
user_bias_mean -0.04765176773071289 and plan_bias_mean -0.5154451727867126 and last_grad_norm 1.1631076335906982 and d_loss 9.055680416526888
current train loss 1872520704.3103027
curren stop limit 5
2 epoch took 6.85s
For k = 0 running on epoch 16, with loss 2743912.9942913055 and lr 0.0005, and best loss of 2484066.200273514
user_emb_mean 0.0 and user_emb_max 0.0 and plan_emb_mean 0.0 and plan_emb_max 0.0
user_bias_mean -0.05858048051595688 and plan_bias_mean -0.5420982241630554 and last_grad_norm 1.149765968322754 and d_loss 10.460542234711005
current train loss 1845531388.7756348
curren stop limit 6
2 epoch took 10.56s
For k = 1 running on epoch 8, with loss 2595689.048713684 and lr 0.001, and best loss of 2595689.048713684
user_emb_mean 0.6643651127815247 and user_emb_max 0.9725829362869263 and plan_emb_mean 0.49451956152915955 and plan_emb_max 0.7076928019523621
user_bias_mean 0.01726648025214672 and plan_bias_mean -0.32083773612976074 and last_grad_norm 1.4138617515563965 and d_loss 0.0
current train loss 2213026819.276123
curren stop limit 0
2 epoch took 5.19s
For k = 0 running on epoch 18, with loss 2779390.0841026306 and lr 0.0005, and best loss of 2484066.200273514
user_emb_mean 0.0 and user_emb_max 0.0 and plan_emb_mean 0.0 and plan_emb_max 0.0
user_bias_mean -0.06947827339172363 and plan_bias_mean -0.5670316219329834 and last_grad_norm 1.1404728889465332 and d_loss 11.888728400096564
current train loss 1825642262.150879
curren stop limit 7
k = 0, converged at epoch 6, last epoch is 20, and best loss of the run is 2484066.20
Run took 118.54716300964355s
2 epoch took 9.99s
For k = 1 running on epoch 10, with loss 2554386.6146450043 and lr 0.001, and best loss of 2554386.6146450043
user_emb_mean 0.6556563973426819 and user_emb_max 0.9722115397453308 and plan_emb_mean 0.4558107554912567 and plan_emb_max 0.672358512878418
user_bias_mean -0.005331635940819979 and plan_bias_mean -0.4004496932029724 and last_grad_norm 1.3485260009765625 and d_loss 0.0
current train loss 2130072713.491211
curren stop limit 0
2 epoch took 9.75s
For k = 1 running on epoch 12, with loss 2554444.1513404846 and lr 0.001, and best loss of 2554386.6146450043
user_emb_mean 0.6474135518074036 and user_emb_max 0.9718402624130249 and plan_emb_mean 0.42088794708251953 and plan_emb_max 0.6685868501663208
user_bias_mean -0.027890212833881378 and plan_bias_mean -0.47088900208473206 and last_grad_norm 1.2970778942108154 and d_loss 0.0022524662144122157
current train loss 2057023215.6384277
curren stop limit 1
2 epoch took 10.15s
For k = 2 running on epoch 2, with loss 3660476.754131317 and lr 0.001, and best loss of 3660476.754131317
user_emb_mean 0.6936231851577759 and user_emb_max 0.9737011194229126 and plan_emb_mean 0.6550753116607666 and plan_emb_max 0.8994306325912476
user_bias_mean 0.08462096750736237 and plan_bias_mean -0.020499058067798615 and last_grad_norm 1.7516714334487915 and d_loss 0.0
current train loss 2676709962.7124023
curren stop limit 0
2 epoch took 10.34s
For k = 1 running on epoch 14, with loss 2582190.7756080627 and lr 0.001, and best loss of 2554386.6146450043
user_emb_mean 0.6396064162254333 and user_emb_max 0.9714690446853638 and plan_emb_mean 0.3893972933292389 and plan_emb_max 0.665951132774353
user_bias_mean -0.050378017127513885 and plan_bias_mean -0.5330051183700562 and last_grad_norm 1.2587981224060059 and d_loss 1.0884867938020633
current train loss 1992468564.888916
curren stop limit 2
2 epoch took 10.29s
For k = 2 running on epoch 4, with loss 3196932.2461719513 and lr 0.001, and best loss of 3196932.2461719513
user_emb_mean 0.6834098696708679 and user_emb_max 0.9733272194862366 and plan_emb_mean 0.6015602350234985 and plan_emb_max 0.830716073513031
user_bias_mean 0.0623195506632328 and plan_bias_mean -0.13340024650096893 and last_grad_norm 1.6312108039855957 and d_loss 0.0
current train loss 2533803732.5097656
curren stop limit 0
2 epoch took 10.40s
For k = 1 running on epoch 16, with loss 2628975.7213802338 and lr 0.001, and best loss of 2554386.6146450043
user_emb_mean 0.6322017908096313 and user_emb_max 0.9710991382598877 and plan_emb_mean 0.3609793484210968 and plan_emb_max 0.6628137826919556
user_bias_mean -0.07278050482273102 and plan_bias_mean -0.5876624584197998 and last_grad_norm 1.2285902500152588 and d_loss 2.9200398368668834
current train loss 1935196184.576416
curren stop limit 3
2 epoch took 11.33s
For k = 2 running on epoch 6, with loss 2900981.504333496 and lr 0.001, and best loss of 2900981.504333496
user_emb_mean 0.6736615300178528 and user_emb_max 0.9729545712471008 and plan_emb_mean 0.5527909398078918 and plan_emb_max 0.7666643261909485
user_bias_mean 0.039752840995788574 and plan_bias_mean -0.23449048399925232 and last_grad_norm 1.5453119277954102 and d_loss 0.0
current train loss 2409536600.708008
curren stop limit 0
2 epoch took 9.77s
For k = 1 running on epoch 18, with loss 2687825.0315589905 and lr 0.001, and best loss of 2554386.6146450043
user_emb_mean 0.6251684427261353 and user_emb_max 0.9707303047180176 and plan_emb_mean 0.33530983328819275 and plan_emb_max 0.6588819622993469
user_bias_mean -0.09508565068244934 and plan_bias_mean -0.6356068849563599 and last_grad_norm 1.1948081254959106 and d_loss 5.22389274000055
current train loss 1884214105.6689453
curren stop limit 4
2 epoch took 10.38s
For k = 2 running on epoch 8, with loss 2723111.7915668488 and lr 0.001, and best loss of 2723111.7915668488
user_emb_mean 0.664411723613739 and user_emb_max 0.9725831747055054 and plan_emb_mean 0.5085357427597046 and plan_emb_max 0.7072690725326538
user_bias_mean 0.017095381394028664 and plan_bias_mean -0.3246418833732605 and last_grad_norm 1.4656234979629517 and d_loss 0.0
current train loss 2301340055.7458496
curren stop limit 0
2 epoch took 9.77s
For k = 1 running on epoch 20, with loss 2753650.2960529327 and lr 0.0005, and best loss of 2554386.6146450043
user_emb_mean 0.6184781789779663 and user_emb_max 0.9703615307807922 and plan_emb_mean 0.31209641695022583 and plan_emb_max 0.6547836661338806
user_bias_mean -0.11728660762310028 and plan_bias_mean -0.6775493621826172 and last_grad_norm 1.1661666631698608 and d_loss 7.800842686283067
current train loss 1838580406.6601562
curren stop limit 5
2 epoch took 10.18s
For k = 2 running on epoch 10, with loss 2627581.972869873 and lr 0.001, and best loss of 2627581.972869873
user_emb_mean 0.6556491851806641 and user_emb_max 0.9722117781639099 and plan_emb_mean 0.46849700808525085 and plan_emb_max 0.6852102279663086
user_bias_mean -0.005574034992605448 and plan_bias_mean -0.4047272205352783 and last_grad_norm 1.3954211473464966 and d_loss 0.0
current train loss 2206838696.895752
curren stop limit 0
2 epoch took 9.83s
For k = 1 running on epoch 22, with loss 2787927.3325748444 and lr 0.0005, and best loss of 2554386.6146450043
user_emb_mean 0.6152289509773254 and user_emb_max 0.9701772332191467 and plan_emb_mean 0.30125555396080017 and plan_emb_max 0.6526432633399963
user_bias_mean -0.12833470106124878 and plan_bias_mean -0.6963728070259094 and last_grad_norm 1.1545851230621338 and d_loss 9.142731824183803
current train loss 1811869526.7248535
curren stop limit 6
2 epoch took 10.33s
For k = 2 running on epoch 12, with loss 2589693.494670868 and lr 0.001, and best loss of 2589693.494670868
user_emb_mean 0.6473503708839417 and user_emb_max 0.971840500831604 and plan_emb_mean 0.4323228597640991 and plan_emb_max 0.6793478727340698
user_bias_mean -0.028207821771502495 and plan_bias_mean -0.4756636321544647 and last_grad_norm 1.3390202522277832 and d_loss 0.0
current train loss 2123989017.9980469
curren stop limit 0
2 epoch took 9.82s
For k = 1 running on epoch 24, with loss 2823646.668926239 and lr 0.0005, and best loss of 2554386.6146450043
user_emb_mean 0.6119964718818665 and user_emb_max 0.9699928760528564 and plan_emb_mean 0.2907518148422241 and plan_emb_max 0.6498021483421326
user_bias_mean -0.13933247327804565 and plan_bias_mean -0.7138086557388306 and last_grad_norm 1.1393769979476929 and d_loss 10.541084608629424
current train loss 1792245835.5493164
curren stop limit 7
2 epoch took 10.27s
For k = 2 running on epoch 14, with loss 2591708.8430423737 and lr 0.001, and best loss of 2589693.494670868
user_emb_mean 0.6394867300987244 and user_emb_max 0.9714692831039429 and plan_emb_mean 0.3996671140193939 and plan_emb_max 0.6762319803237915
user_bias_mean -0.05077163502573967 and plan_bias_mean -0.5382770299911499 and last_grad_norm 1.2952734231948853 and d_loss 0.07782188802083986
current train loss 2051118243.9074707
curren stop limit 1
k = 1, converged at epoch 12, last epoch is 26, and best loss of the run is 2554386.61
Run took 277.85090041160583s
2 epoch took 10.66s
For k = 2 running on epoch 16, with loss 2620048.043958664 and lr 0.001, and best loss of 2589693.494670868
user_emb_mean 0.6320284008979797 and user_emb_max 0.9710993766784668 and plan_emb_mean 0.37019672989845276 and plan_emb_max 0.6753045320510864
user_bias_mean -0.07324675470590591 and plan_bias_mean -0.5933552980422974 and last_grad_norm 1.254308819770813 and d_loss 1.1721290318819708
current train loss 1986727081.488037
curren stop limit 2
2 epoch took 17.65s
For k = 3 running on epoch 2, with loss 3954852.196369171 and lr 0.001, and best loss of 3954852.196369171
user_emb_mean 0.6935931444168091 and user_emb_max 0.9737021923065186 and plan_emb_mean 0.6343855857849121 and plan_emb_max 0.9007015228271484
user_bias_mean 0.08460600674152374 and plan_bias_mean -0.03354122117161751 and last_grad_norm 1.7978578805923462 and d_loss 0.0
current train loss 2752690400.8276367
curren stop limit 0
2 epoch took 11.74s
For k = 2 running on epoch 18, with loss 2666376.0367794037 and lr 0.001, and best loss of 2589693.494670868
user_emb_mean 0.6249436736106873 and user_emb_max 0.9707305431365967 and plan_emb_mean 0.34357526898384094 and plan_emb_max 0.674938976764679
user_bias_mean -0.09562143683433533 and plan_bias_mean -0.6416766047477722 and last_grad_norm 1.2220654487609863 and d_loss 2.9610663295225823
current train loss 1929649236.9006348
curren stop limit 3
2 epoch took 10.83s
For k = 3 running on epoch 4, with loss 3407668.706794739 and lr 0.001, and best loss of 3407668.706794739
user_emb_mean 0.6833805441856384 and user_emb_max 0.9790248274803162 and plan_emb_mean 0.5821640491485596 and plan_emb_max 0.8323841691017151
user_bias_mean 0.06229359284043312 and plan_bias_mean -0.14499948918819427 and last_grad_norm 1.684897541999817 and d_loss 0.0
current train loss 2599098511.6625977
curren stop limit 0
2 epoch took 10.49s
For k = 2 running on epoch 20, with loss 2723686.6538829803 and lr 0.001, and best loss of 2589693.494670868
user_emb_mean 0.618204653263092 and user_emb_max 0.9703618288040161 and plan_emb_mean 0.3195068836212158 and plan_emb_max 0.6747403144836426
user_bias_mean -0.11788660287857056 and plan_bias_mean -0.6839175820350647 and last_grad_norm 1.1896713972091675 and d_loss 5.174093362316687
current train loss 1878798457.3120117
curren stop limit 4
2 epoch took 10.24s
For k = 2 running on epoch 22, with loss 2788011.912103653 and lr 0.0005, and best loss of 2589693.494670868
user_emb_mean 0.6117817759513855 and user_emb_max 0.9699931144714355 and plan_emb_mean 0.2977079451084137 and plan_emb_max 0.6741556525230408
user_bias_mean -0.14004398882389069 and plan_bias_mean -0.7207546234130859 and last_grad_norm 1.1635894775390625 and d_loss 7.657988014446085
current train loss 1833328614.8583984
curren stop limit 5
2 epoch took 11.13s
For k = 3 running on epoch 6, with loss 3045809.785139084 and lr 0.001, and best loss of 3045809.785139084
user_emb_mean 0.6736291646957397 and user_emb_max 0.9816381335258484 and plan_emb_mean 0.5346196889877319 and plan_emb_max 0.7685902118682861
user_bias_mean 0.039705876260995865 and plan_bias_mean -0.2448398768901825 and last_grad_norm 1.5805965662002563 and d_loss 0.0
current train loss 2465870649.156494
curren stop limit 0
2 epoch took 10.09s
For k = 2 running on epoch 24, with loss 2822168.6879348755 and lr 0.0005, and best loss of 2589693.494670868
user_emb_mean 0.6086584329605103 and user_emb_max 0.96980881690979 and plan_emb_mean 0.2875119149684906 and plan_emb_max 0.6733850240707397
user_bias_mean -0.1510685682296753 and plan_bias_mean -0.7372500896453857 and last_grad_norm 1.15113365650177 and d_loss 8.976938535096933
current train loss 1806674659.3115234
curren stop limit 6
2 epoch took 11.17s
For k = 3 running on epoch 8, with loss 2816027.190223694 and lr 0.001, and best loss of 2816027.190223694
user_emb_mean 0.6643713116645813 and user_emb_max 0.9875379800796509 and plan_emb_mean 0.49150413274765015 and plan_emb_max 0.7141206860542297
user_bias_mean 0.017012163996696472 and plan_bias_mean -0.3339451551437378 and last_grad_norm 1.4982810020446777 and d_loss 0.0
current train loss 2350068176.895752
curren stop limit 0
2 epoch took 11.12s
For k = 2 running on epoch 26, with loss 2857101.761943817 and lr 0.0005, and best loss of 2589693.494670868
user_emb_mean 0.6055470705032349 and user_emb_max 0.9696245789527893 and plan_emb_mean 0.27762937545776367 and plan_emb_max 0.6728256940841675
user_bias_mean -0.16204333305358887 and plan_bias_mean -0.7524595260620117 and last_grad_norm 1.142609715461731 and d_loss 10.325865505830254
current train loss 1787070283.161621
curren stop limit 7
2 epoch took 11.74s
For k = 3 running on epoch 10, with loss 2681097.9711647034 and lr 0.001, and best loss of 2681097.9711647034
user_emb_mean 0.6555958986282349 and user_emb_max 0.9973140358924866 and plan_emb_mean 0.45249855518341064 and plan_emb_max 0.6688355207443237
user_bias_mean -0.005707663483917713 and plan_bias_mean -0.4132230877876282 and last_grad_norm 1.4263660907745361 and d_loss 0.0
current train loss 2249064051.604004
curren stop limit 0
k = 2, converged at epoch 14, last epoch is 28, and best loss of the run is 2589693.49
Run took 296.0886504650116s
2 epoch took 11.06s
For k = 3 running on epoch 12, with loss 2613174.784767151 and lr 0.001, and best loss of 2613174.784767151
user_emb_mean 0.6472821235656738 and user_emb_max 1.0097712278366089 and plan_emb_mean 0.4172798991203308 and plan_emb_max 0.6285983920097351
user_bias_mean -0.028400400653481483 and plan_bias_mean -0.4834907650947571 and last_grad_norm 1.36321222782135 and d_loss 0.0
current train loss 2160699137.1875
curren stop limit 0
2 epoch took 11.69s
For k = 4 running on epoch 2, with loss 4323890.793460846 and lr 0.001, and best loss of 4323890.793460846
user_emb_mean 0.6936106085777283 and user_emb_max 0.9737027287483215 and plan_emb_mean 0.6362054944038391 and plan_emb_max 0.900633692741394
user_bias_mean 0.08460971713066101 and plan_bias_mean -0.01838614232838154 and last_grad_norm 1.8631484508514404 and d_loss 0.0
current train loss 2850966502.51709
curren stop limit 0
2 epoch took 10.43s
For k = 3 running on epoch 14, with loss 2593447.0452651978 and lr 0.001, and best loss of 2593447.0452651978
user_emb_mean 0.6394014954566956 and user_emb_max 1.0213905572891235 and plan_emb_mean 0.38549768924713135 and plan_emb_max 0.5928533673286438
user_bias_mean -0.051032520830631256 and plan_bias_mean -0.5455970168113708 and last_grad_norm 1.3177776336669922 and d_loss 0.0
current train loss 2083120605.0195312
curren stop limit 0
2 epoch took 11.95s
For k = 3 running on epoch 16, with loss 2605891.0152168274 and lr 0.001, and best loss of 2593447.0452651978
user_emb_mean 0.6319248676300049 and user_emb_max 1.0376865863800049 and plan_emb_mean 0.3568268120288849 and plan_emb_max 0.561050295829773
user_bias_mean -0.07357994467020035 and plan_bias_mean -0.6002727150917053 and last_grad_norm 1.2734512090682983 and d_loss 0.47982356047517283
current train loss 2014721182.7355957
curren stop limit 1
2 epoch took 13.26s
For k = 4 running on epoch 4, with loss 3644899.8117980957 and lr 0.001, and best loss of 3644899.8117980957
user_emb_mean 0.683393120765686 and user_emb_max 0.9733288884162903 and plan_emb_mean 0.5837409496307373 and plan_emb_max 0.8344011902809143
user_bias_mean 0.06225641816854477 and plan_bias_mean -0.13191190361976624 and last_grad_norm 1.7413725852966309 and d_loss 0.0
current train loss 2682737634.2822266
curren stop limit 0
2 epoch took 10.85s
For k = 3 running on epoch 18, with loss 2641867.2148952484 and lr 0.001, and best loss of 2593447.0452651978
user_emb_mean 0.6248218417167664 and user_emb_max 1.0435256958007812 and plan_emb_mean 0.3309354782104492 and plan_emb_max 0.5326390266418457
user_bias_mean -0.09602970629930496 and plan_bias_mean -0.6482877135276794 and last_grad_norm 1.2420443296432495 and d_loss 1.8670197919965381
current train loss 1954191691.8652344
curren stop limit 2
2 epoch took 11.63s
For k = 4 running on epoch 6, with loss 3190348.111448288 and lr 0.001, and best loss of 3190348.111448288
user_emb_mean 0.6736330389976501 and user_emb_max 0.9729562401771545 and plan_emb_mean 0.5358991026878357 and plan_emb_max 0.789178192615509
user_bias_mean 0.039609767496585846 and plan_bias_mean -0.23384042084217072 and last_grad_norm 1.6323297023773193 and d_loss 0.0
current train loss 2537168089.82666
curren stop limit 0
2 epoch took 11.10s
For k = 3 running on epoch 20, with loss 2692091.308965683 and lr 0.001, and best loss of 2593447.0452651978
user_emb_mean 0.6180649399757385 and user_emb_max 1.0616419315338135 and plan_emb_mean 0.3075394928455353 and plan_emb_max 0.5124181509017944
user_bias_mean -0.1183684915304184 and plan_bias_mean -0.6902824640274048 and last_grad_norm 1.2042330503463745 and d_loss 3.803596602466899
current train loss 1900388515.2075195
curren stop limit 3
2 epoch took 11.92s
For k = 4 running on epoch 8, with loss 2899163.220293045 and lr 0.001, and best loss of 2899163.220293045
user_emb_mean 0.6643652319908142 and user_emb_max 0.9725847840309143 and plan_emb_mean 0.49245956540107727 and plan_emb_max 0.7549623847007751
user_bias_mean 0.0168454647064209 and plan_bias_mean -0.3250252306461334 and last_grad_norm 1.5341877937316895 and d_loss 0.0
current train loss 2411018299.3884277
curren stop limit 0
2 epoch took 11.02s
For k = 3 running on epoch 22, with loss 2751773.143201828 and lr 0.001, and best loss of 2593447.0452651978
user_emb_mean 0.6116254925727844 and user_emb_max 1.0703812837600708 and plan_emb_mean 0.28636497259140015 and plan_emb_max 0.5120335817337036
user_bias_mean -0.14059749245643616 and plan_bias_mean -0.7269100546836853 and last_grad_norm 1.1754536628723145 and d_loss 6.104851773460457
current train loss 1852368636.1425781
curren stop limit 4
2 epoch took 11.87s
For k = 4 running on epoch 10, with loss 2724644.223552704 and lr 0.001, and best loss of 2724644.223552704
user_emb_mean 0.6555812954902649 and user_emb_max 0.9722133874893188 and plan_emb_mean 0.4531407356262207 and plan_emb_max 0.7297854423522949
user_bias_mean -0.00594877265393734 and plan_bias_mean -0.40628373622894287 and last_grad_norm 1.4579921960830688 and d_loss 0.0
current train loss 2301350317.1362305
curren stop limit 0
2 epoch took 10.51s
For k = 3 running on epoch 24, with loss 2817441.5778598785 and lr 0.0005, and best loss of 2593447.0452651978
user_emb_mean 0.6054773330688477 and user_emb_max 1.08376944065094 and plan_emb_mean 0.26716378331184387 and plan_emb_max 0.5105676054954529
user_bias_mean -0.1627143770456314 and plan_bias_mean -0.7587525844573975 and last_grad_norm 1.1492862701416016 and d_loss 8.6369425974447
current train loss 1809320842.680664
curren stop limit 5
2 epoch took 11.00s
For k = 4 running on epoch 12, with loss 2632035.5287456512 and lr 0.001, and best loss of 2632035.5287456512
user_emb_mean 0.6472612023353577 and user_emb_max 0.9778171181678772 and plan_emb_mean 0.4176364541053772 and plan_emb_max 0.7118965983390808
user_bias_mean -0.028715629130601883 and plan_bias_mean -0.47839435935020447 and last_grad_norm 1.390677809715271 and d_loss 0.0
current train loss 2205720844.202881
curren stop limit 0
2 epoch took 10.71s
For k = 3 running on epoch 26, with loss 2851514.344766617 and lr 0.0005, and best loss of 2593447.0452651978
user_emb_mean 0.6024827361106873 and user_emb_max 1.086974024772644 and plan_emb_mean 0.25817421078681946 and plan_emb_max 0.5097631216049194
user_bias_mean -0.17371807992458344 and plan_bias_mean -0.7729491591453552 and last_grad_norm 1.1419998407363892 and d_loss 9.950744896548674
current train loss 1784038832.006836
curren stop limit 6
2 epoch took 11.67s
For k = 4 running on epoch 14, with loss 2596051.2346954346 and lr 0.001, and best loss of 2596051.2346954346
user_emb_mean 0.6393780708312988 and user_emb_max 0.9914689064025879 and plan_emb_mean 0.38561558723449707 and plan_emb_max 0.7001944780349731
user_bias_mean -0.05141855403780937 and plan_bias_mean -0.5421338677406311 and last_grad_norm 1.3388495445251465 and d_loss 0.0
current train loss 2122005151.241455
curren stop limit 0
2 epoch took 10.76s
For k = 3 running on epoch 28, with loss 2886465.0754528046 and lr 0.0005, and best loss of 2593447.0452651978
user_emb_mean 0.599496066570282 and user_emb_max 1.0930036306381226 and plan_emb_mean 0.2494489550590515 and plan_emb_max 0.5090824961662292
user_bias_mean -0.1846718043088913 and plan_bias_mean -0.7859988808631897 and last_grad_norm 1.1307711601257324 and d_loss 11.298400355718222
current train loss 1765395594.4213867
curren stop limit 7
2 epoch took 11.82s
For k = 4 running on epoch 16, with loss 2598830.184228897 and lr 0.001, and best loss of 2596051.2346954346
user_emb_mean 0.6319032907485962 and user_emb_max 1.0003610849380493 and plan_emb_mean 0.3567485213279724 and plan_emb_max 0.6926736235618591
user_bias_mean -0.07403071224689484 and plan_bias_mean -0.5982550978660583 and last_grad_norm 1.2956223487854004 and d_loss 0.10704524996743939
current train loss 2048461649.4238281
curren stop limit 1
k = 3, converged at epoch 16, last epoch is 30, and best loss of the run is 2593447.05
Run took 335.46157336235046s
2 epoch took 12.32s
For k = 4 running on epoch 18, with loss 2627544.3319664 and lr 0.001, and best loss of 2596051.2346954346
user_emb_mean 0.6248073577880859 and user_emb_max 1.0146234035491943 and plan_emb_mean 0.3307165205478668 and plan_emb_max 0.688135027885437
user_bias_mean -0.0965360775589943 and plan_bias_mean -0.6474815607070923 and last_grad_norm 1.2508870363235474 and d_loss 1.2131153981119447
current train loss 1983562388.9868164
curren stop limit 2
2 epoch took 12.97s
For k = 6 running on epoch 2, with loss 5434180.9373931885 and lr 0.001, and best loss of 5434180.9373931885
user_emb_mean 0.6936664581298828 and user_emb_max 0.9737028479576111 and plan_emb_mean 0.6410322785377502 and plan_emb_max 0.9004167318344116
user_bias_mean 0.08464422076940536 and plan_bias_mean -0.023986728861927986 and last_grad_norm 1.9953725337982178 and d_loss 0.0
current train loss 3011110519.0942383
curren stop limit 0
2 epoch took 12.45s
For k = 4 running on epoch 20, with loss 2673895.8296375275 and lr 0.001, and best loss of 2596051.2346954346
user_emb_mean 0.6180601119995117 and user_emb_max 1.0259802341461182 and plan_emb_mean 0.30721303820610046 and plan_emb_max 0.6858136653900146
user_bias_mean -0.11892828345298767 and plan_bias_mean -0.6905390620231628 and last_grad_norm 1.2183408737182617 and d_loss 2.998576988840728
current train loss 1926065337.7331543
curren stop limit 3
2 epoch took 13.65s
For k = 6 running on epoch 4, with loss 4464518.758026123 and lr 0.001, and best loss of 4464518.758026123
user_emb_mean 0.6834348440170288 and user_emb_max 0.9733290076255798 and plan_emb_mean 0.5881297588348389 and plan_emb_max 0.8322412967681885
user_bias_mean 0.06225387006998062 and plan_bias_mean -0.1371479481458664 and last_grad_norm 1.8628484010696411 and d_loss 0.0
current train loss 2819120198.898926
curren stop limit 0
2 epoch took 11.61s
For k = 4 running on epoch 22, with loss 2731535.849046707 and lr 0.001, and best loss of 2596051.2346954346
user_emb_mean 0.6116333603858948 and user_emb_max 1.037950038909912 and plan_emb_mean 0.28596431016921997 and plan_emb_max 0.6846380829811096
user_bias_mean -0.14120343327522278 and plan_bias_mean -0.7280601263046265 and last_grad_norm 1.1862472295761108 and d_loss 5.218872899754903
current train loss 1874918367.8295898
curren stop limit 4
2 epoch took 13.50s
For k = 6 running on epoch 6, with loss 3785554.2380008698 and lr 0.001, and best loss of 3785554.2380008698
user_emb_mean 0.6736451983451843 and user_emb_max 0.9729564189910889 and plan_emb_mean 0.5397774577140808 and plan_emb_max 0.7691491842269897
user_bias_mean 0.03953683748841286 and plan_bias_mean -0.2389945387840271 and last_grad_norm 1.7275829315185547 and d_loss 0.0
current train loss 2653440452.9248047
curren stop limit 0
2 epoch took 13.08s
For k = 4 running on epoch 24, with loss 2795996.46931839 and lr 0.0005, and best loss of 2596051.2346954346
user_emb_mean 0.6055018305778503 and user_emb_max 1.044527292251587 and plan_emb_mean 0.26696160435676575 and plan_emb_max 0.6842035055160522
user_bias_mean -0.1633594036102295 and plan_bias_mean -0.7606508135795593 and last_grad_norm 1.1637994050979614 and d_loss 7.701898635541091
current train loss 1829175908.774414
curren stop limit 5
2 epoch took 13.40s
For k = 6 running on epoch 8, with loss 3322262.8242168427 and lr 0.001, and best loss of 3322262.8242168427
user_emb_mean 0.6643348932266235 and user_emb_max 0.9725849032402039 and plan_emb_mean 0.4957858622074127 and plan_emb_max 0.7108162045478821
user_bias_mean 0.01667102426290512 and plan_bias_mean -0.33033087849617004 and last_grad_norm 1.6230682134628296 and d_loss 0.0
current train loss 2510305451.982422
curren stop limit 0
2 epoch took 13.04s
For k = 4 running on epoch 26, with loss 2829691.373184204 and lr 0.0005, and best loss of 2596051.2346954346
user_emb_mean 0.6025145649909973 and user_emb_max 1.0473806858062744 and plan_emb_mean 0.25833770632743835 and plan_emb_max 0.6844448447227478
user_bias_mean -0.17437992990016937 and plan_bias_mean -0.7751652598381042 and last_grad_norm 1.1509839296340942 and d_loss 8.999827713961889
current train loss 1802493385.5737305
curren stop limit 6
2 epoch took 12.74s
For k = 6 running on epoch 10, with loss 3016624.904182434 and lr 0.001, and best loss of 3016624.904182434
user_emb_mean 0.6554993391036987 and user_emb_max 0.9722135663032532 and plan_emb_mean 0.4559054672718048 and plan_emb_max 0.6683321595191956
user_bias_mean -0.006252513732761145 and plan_bias_mean -0.41191476583480835 and last_grad_norm 1.535772442817688 and d_loss 0.0
current train loss 2386276102.1154785
curren stop limit 0
2 epoch took 11.52s
For k = 4 running on epoch 28, with loss 2864640.4852485657 and lr 0.0005, and best loss of 2596051.2346954346
user_emb_mean 0.5995320677757263 and user_emb_max 1.054070234298706 and plan_emb_mean 0.25029000639915466 and plan_emb_max 0.6845036745071411
user_bias_mean -0.18534818291664124 and plan_bias_mean -0.7884957194328308 and last_grad_norm 1.135788083076477 and d_loss 10.34606894361396
current train loss 1782979490.8361816
curren stop limit 7
2 epoch took 12.26s
For k = 6 running on epoch 12, with loss 2824964.7686100006 and lr 0.001, and best loss of 2824964.7686100006
user_emb_mean 0.6471207141876221 and user_emb_max 0.9718422293663025 and plan_emb_mean 0.41984495520591736 and plan_emb_max 0.632497251033783
user_bias_mean -0.029173800721764565 and plan_bias_mean -0.4844930171966553 and last_grad_norm 1.4582680463790894 and d_loss 0.0
current train loss 2278452108.276367
curren stop limit 0
k = 4, converged at epoch 16, last epoch is 30, and best loss of the run is 2596051.23
Run took 359.5251076221466s
2 epoch took 12.63s
For k = 6 running on epoch 14, with loss 2715129.001815796 and lr 0.001, and best loss of 2715129.001815796
user_emb_mean 0.6391752362251282 and user_emb_max 0.9714710116386414 and plan_emb_mean 0.38729000091552734 and plan_emb_max 0.6014821529388428
user_bias_mean -0.05205041915178299 and plan_bias_mean -0.5487995147705078 and last_grad_norm 1.3937466144561768 and d_loss 0.0
current train loss 2184431502.4816895
curren stop limit 0
2 epoch took 13.97s
For k = 8 running on epoch 2, with loss 6151054.302997589 and lr 0.001, and best loss of 6151054.302997589
user_emb_mean 0.6936078071594238 and user_emb_max 0.9737029671669006 and plan_emb_mean 0.6479610800743103 and plan_emb_max 0.9008590579032898
user_bias_mean 0.08456173539161682 and plan_bias_mean -0.02687987871468067 and last_grad_norm 2.089005470275879 and d_loss 0.0
current train loss 3150428210.4492188
curren stop limit 0
2 epoch took 12.63s
For k = 6 running on epoch 16, with loss 2663248.052061081 and lr 0.001, and best loss of 2663248.052061081
user_emb_mean 0.6316372156143188 and user_emb_max 0.9711011052131653 and plan_emb_mean 0.35792282223701477 and plan_emb_max 0.5747840404510498
user_bias_mean -0.07484937459230423 and plan_bias_mean -0.6055423021316528 and last_grad_norm 1.333156943321228 and d_loss 0.0
current train loss 2102096376.1328125
curren stop limit 0
2 epoch took 13.94s
For k = 8 running on epoch 4, with loss 4984093.732021332 and lr 0.001, and best loss of 4984093.732021332
user_emb_mean 0.6833672523498535 and user_emb_max 0.9733291268348694 and plan_emb_mean 0.5945769548416138 and plan_emb_max 0.8335254192352295
user_bias_mean 0.06215003505349159 and plan_bias_mean -0.1398380845785141 and last_grad_norm 1.9335589408874512 and d_loss 0.0
current train loss 2938321334.074707
curren stop limit 0
2 epoch took 12.89s
For k = 6 running on epoch 18, with loss 2652234.657503128 and lr 0.001, and best loss of 2652234.657503128
user_emb_mean 0.6244785189628601 and user_emb_max 0.9707322716712952 and plan_emb_mean 0.33153247833251953 and plan_emb_max 0.5517032742500305
user_bias_mean -0.09755054861307144 and plan_bias_mean -0.6554088592529297 and last_grad_norm 1.2922800779342651 and d_loss 0.0
current train loss 2029735112.5195312
curren stop limit 0
2 epoch took 14.16s
For k = 8 running on epoch 6, with loss 4155425.0496902466 and lr 0.001, and best loss of 4155425.0496902466
user_emb_mean 0.6735602021217346 and user_emb_max 0.9729564785957336 and plan_emb_mean 0.5456975102424622 and plan_emb_max 0.775479257106781
user_bias_mean 0.03939888998866081 and plan_bias_mean -0.2416285276412964 and last_grad_norm 1.7990875244140625 and d_loss 0.0
current train loss 2755656175.275879
curren stop limit 0
2 epoch took 13.11s
For k = 6 running on epoch 20, with loss 2669123.3027000427 and lr 0.001, and best loss of 2652234.657503128
user_emb_mean 0.6176697015762329 and user_emb_max 0.9703635573387146 and plan_emb_mean 0.3086082935333252 and plan_emb_max 0.5490323305130005
user_bias_mean -0.12014083564281464 and plan_bias_mean -0.6990503072738647 and last_grad_norm 1.254957675933838 and d_loss 0.6367703984689655
current train loss 1965955850.9533691
curren stop limit 1
2 epoch took 14.13s
For k = 8 running on epoch 8, with loss 3579425.6905937195 and lr 0.001, and best loss of 3579425.6905937195
user_emb_mean 0.6642259359359741 and user_emb_max 0.9725850224494934 and plan_emb_mean 0.5011657476425171 and plan_emb_max 0.723622739315033
user_bias_mean 0.016486844047904015 and plan_bias_mean -0.3330090343952179 and last_grad_norm 1.676466941833496 and d_loss 0.0
current train loss 2598130630.632324
curren stop limit 0
2 epoch took 12.88s
For k = 6 running on epoch 22, with loss 2703644.6256866455 and lr 0.001, and best loss of 2652234.657503128
user_emb_mean 0.6111685633659363 and user_emb_max 0.9699949026107788 and plan_emb_mean 0.29115745425224304 and plan_emb_max 0.54780113697052
user_bias_mean -0.14260928332805634 and plan_bias_mean -0.7370871901512146 and last_grad_norm 1.2170765399932861 and d_loss 1.9383642408140433
current train loss 1910045692.0422363
curren stop limit 2
2 epoch took 14.09s
For k = 8 running on epoch 10, with loss 3190448.0104808807 and lr 0.001, and best loss of 3190448.0104808807
user_emb_mean 0.6553594470024109 and user_emb_max 0.9722136855125427 and plan_emb_mean 0.4607287049293518 and plan_emb_max 0.6793283224105835
user_bias_mean -0.006498381961137056 and plan_bias_mean -0.41477257013320923 and last_grad_norm 1.5874959230422974 and d_loss 0.0
current train loss 2461933438.5595703
curren stop limit 0
2 epoch took 12.69s
For k = 6 running on epoch 24, with loss 2750662.0435886383 and lr 0.001, and best loss of 2652234.657503128
user_emb_mean 0.6048895120620728 and user_emb_max 0.9696263074874878 and plan_emb_mean 0.280085027217865 and plan_emb_max 0.5470624566078186
user_bias_mean -0.1649516075849533 and plan_bias_mean -0.770121157169342 and last_grad_norm 1.1845656633377075 and d_loss 3.711111526540866
current train loss 1862383022.9101562
curren stop limit 3
2 epoch took 14.02s
For k = 8 running on epoch 12, with loss 2938065.7638454437 and lr 0.001, and best loss of 2938065.7638454437
user_emb_mean 0.6469456553459167 and user_emb_max 0.971842348575592 and plan_emb_mean 0.4241165220737457 and plan_emb_max 0.659161388874054
user_bias_mean -0.0294917244464159 and plan_bias_mean -0.48762035369873047 and last_grad_norm 1.4936012029647827 and d_loss 0.0
current train loss 2343850934.284668
curren stop limit 0
2 epoch took 12.96s
For k = 6 running on epoch 26, with loss 2802019.721342087 and lr 0.001, and best loss of 2652234.657503128
user_emb_mean 0.598720133304596 and user_emb_max 0.9756438732147217 and plan_emb_mean 0.2742387354373932 and plan_emb_max 0.5469346046447754
user_bias_mean -0.1871599555015564 and plan_bias_mean -0.7986501455307007 and last_grad_norm 1.1674882173538208 and d_loss 5.647504206131206
current train loss 1822752162.74292
curren stop limit 4
2 epoch took 13.90s
For k = 8 running on epoch 14, with loss 2784285.567943573 and lr 0.001, and best loss of 2784285.567943573
user_emb_mean 0.6389636993408203 and user_emb_max 0.9714711308479309 and plan_emb_mean 0.39103421568870544 and plan_emb_max 0.6448159217834473
user_bias_mean -0.05244487151503563 and plan_bias_mean -0.5522381067276001 and last_grad_norm 1.4272558689117432 and d_loss 0.0
current train loss 2241118869.974365
curren stop limit 0
2 epoch took 12.64s
For k = 6 running on epoch 28, with loss 2857457.612268448 and lr 0.0005, and best loss of 2652234.657503128
user_emb_mean 0.5925631523132324 and user_emb_max 0.9879819750785828 and plan_emb_mean 0.2708289921283722 and plan_emb_max 0.5475433468818665
user_bias_mean -0.20923906564712524 and plan_bias_mean -0.8231694102287292 and last_grad_norm 1.142029881477356 and d_loss 7.737737465451914
current train loss 1789462566.8847656
curren stop limit 5
2 epoch took 14.50s
For k = 8 running on epoch 16, with loss 2700944.8618011475 and lr 0.001, and best loss of 2700944.8618011475
user_emb_mean 0.6313880085945129 and user_emb_max 0.9711012244224548 and plan_emb_mean 0.3614486753940582 and plan_emb_max 0.6353346109390259
user_bias_mean -0.07532588392496109 and plan_bias_mean -0.6093058586120605 and last_grad_norm 1.367350459098816 and d_loss 0.0
current train loss 2151392931.0632324
curren stop limit 0
2 epoch took 14.26s
For k = 6 running on epoch 30, with loss 2886821.0446834564 and lr 0.0005, and best loss of 2652234.657503128
user_emb_mean 0.589408278465271 and user_emb_max 0.9935377836227417 and plan_emb_mean 0.2696015536785126 and plan_emb_max 0.547820508480072
user_bias_mean -0.22021569311618805 and plan_bias_mean -0.8339666128158569 and last_grad_norm 1.136228084564209 and d_loss 8.844857920723092
current train loss 1774391017.4804688
curren stop limit 6
2 epoch took 13.96s
For k = 8 running on epoch 18, with loss 2666895.9530715942 and lr 0.001, and best loss of 2666895.9530715942
user_emb_mean 0.6241886019706726 and user_emb_max 0.9707323908805847 and plan_emb_mean 0.3372439742088318 and plan_emb_max 0.6291314363479614
user_bias_mean -0.09810866415500641 and plan_bias_mean -0.6594942808151245 and last_grad_norm 1.3143547773361206 and d_loss 0.0
current train loss 2072986408.1530762
curren stop limit 0
2 epoch took 12.67s
For k = 6 running on epoch 32, with loss 2916986.2369766235 and lr 0.0005, and best loss of 2652234.657503128
user_emb_mean 0.5861876606941223 and user_emb_max 0.9976791143417358 and plan_emb_mean 0.26865383982658386 and plan_emb_max 0.5483700633049011
user_bias_mean -0.2311384230852127 and plan_bias_mean -0.8437647819519043 and last_grad_norm 1.1291481256484985 and d_loss 9.982207974114116
current train loss 1760500375.1391602
curren stop limit 7
k = 6, converged at epoch 20, last epoch is 34, and best loss of the run is 2652234.66
Run took 439.77206563949585s
2 epoch took 14.03s
For k = 8 running on epoch 20, with loss 2664985.541770935 and lr 0.001, and best loss of 2664985.541770935
user_emb_mean 0.6172967553138733 and user_emb_max 0.9703636169433594 and plan_emb_mean 0.32117852568626404 and plan_emb_max 0.6252543926239014
user_bias_mean -0.12077531963586807 and plan_bias_mean -0.7034224271774292 and last_grad_norm 1.2707406282424927 and d_loss 0.0
current train loss 2006014086.6699219
curren stop limit 0
2 epoch took 14.00s
For k = 8 running on epoch 22, with loss 2680853.3458328247 and lr 0.001, and best loss of 2664985.541770935
user_emb_mean 0.6105865240097046 and user_emb_max 0.9699949622154236 and plan_emb_mean 0.3135187327861786 and plan_emb_max 0.623116672039032
user_bias_mean -0.1433107703924179 and plan_bias_mean -0.7416825890541077 and last_grad_norm 1.2411900758743286 and d_loss 0.5954180168401657
current train loss 1951354586.710205
curren stop limit 1
2 epoch took 15.33s
For k = 12 running on epoch 2, with loss 7548872.617084503 and lr 0.001, and best loss of 7548872.617084503
user_emb_mean 0.6935821175575256 and user_emb_max 0.9737027287483215 and plan_emb_mean 0.645916223526001 and plan_emb_max 0.9017054438591003
user_bias_mean 0.08449913561344147 and plan_bias_mean -0.022445352748036385 and last_grad_norm 2.2361834049224854 and d_loss 0.0
current train loss 3383509167.9882812
curren stop limit 0
2 epoch took 13.90s
For k = 8 running on epoch 24, with loss 2712935.5161685944 and lr 0.001, and best loss of 2664985.541770935
user_emb_mean 0.6039161682128906 and user_emb_max 0.9696264266967773 and plan_emb_mean 0.30996426939964294 and plan_emb_max 0.6222916841506958
user_bias_mean -0.16570638120174408 and plan_bias_mean -0.7748583555221558 and last_grad_norm 1.2054128646850586 and d_loss 1.7992583316529216
current train loss 1907814928.2336426
curren stop limit 2
2 epoch took 15.39s
For k = 12 running on epoch 4, with loss 6005584.37204361 and lr 0.001, and best loss of 6005584.37204361
user_emb_mean 0.6833447217941284 and user_emb_max 0.9733288884162903 and plan_emb_mean 0.5925759077072144 and plan_emb_max 0.8357549905776978
user_bias_mean 0.06206011399626732 and plan_bias_mean -0.13608020544052124 and last_grad_norm 2.0342659950256348 and d_loss 0.0
current train loss 3136497486.8945312
curren stop limit 0
2 epoch took 13.80s
For k = 8 running on epoch 26, with loss 2756068.78540802 and lr 0.001, and best loss of 2664985.541770935
user_emb_mean 0.5972185134887695 and user_emb_max 0.9692590832710266 and plan_emb_mean 0.30791887640953064 and plan_emb_max 0.6221461892127991
user_bias_mean -0.18796034157276154 and plan_bias_mean -0.8034706711769104 and last_grad_norm 1.189095139503479 and d_loss 3.417776277186043
current train loss 1871060873.2128906
curren stop limit 3
2 epoch took 15.30s
For k = 12 running on epoch 6, with loss 4887221.928390503 and lr 0.001, and best loss of 4887221.928390503
user_emb_mean 0.6735357046127319 and user_emb_max 0.9729562401771545 and plan_emb_mean 0.5437000393867493 and plan_emb_max 0.7756811380386353
user_bias_mean 0.0392640121281147 and plan_bias_mean -0.23862916231155396 and last_grad_norm 1.901524305343628 and d_loss 0.0
current train loss 2924537428.0932617
curren stop limit 0
2 epoch took 13.94s
For k = 8 running on epoch 28, with loss 2805939.033647537 and lr 0.0005, and best loss of 2664985.541770935
user_emb_mean 0.5904800295829773 and user_emb_max 0.9713881015777588 and plan_emb_mean 0.30630946159362793 and plan_emb_max 0.622387707233429
user_bias_mean -0.21007464826107025 and plan_bias_mean -0.8280247449874878 and last_grad_norm 1.1708601713180542 and d_loss 5.289090303391884
current train loss 1838440646.2097168
curren stop limit 4
2 epoch took 15.32s
For k = 12 running on epoch 8, with loss 4090763.969024658 and lr 0.001, and best loss of 4090763.969024658
user_emb_mean 0.6641918420791626 and user_emb_max 0.9725847840309143 and plan_emb_mean 0.49912139773368835 and plan_emb_max 0.7211849093437195
user_bias_mean 0.01628396101295948 and plan_bias_mean -0.33088794350624084 and last_grad_norm 1.773913860321045 and d_loss 0.0
current train loss 2742442214.6533203
curren stop limit 0
2 epoch took 14.25s
For k = 8 running on epoch 30, with loss 2831169.9219551086 and lr 0.0005, and best loss of 2664985.541770935
user_emb_mean 0.5870227217674255 and user_emb_max 0.9740419983863831 and plan_emb_mean 0.3056895136833191 and plan_emb_max 0.6230443716049194
user_bias_mean -0.22106653451919556 and plan_bias_mean -0.8388000726699829 and last_grad_norm 1.164588451385498 and d_loss 6.235845470056127
current train loss 1825307337.2277832
curren stop limit 5
2 epoch took 15.43s
For k = 12 running on epoch 10, with loss 3535929.073535919 and lr 0.001, and best loss of 3535929.073535919
user_emb_mean 0.6553114056587219 and user_emb_max 0.9722133874893188 and plan_emb_mean 0.45861127972602844 and plan_emb_max 0.6720462441444397
user_bias_mean -0.0067860400304198265 and plan_bias_mean -0.41361159086227417 and last_grad_norm 1.6522507667541504 and d_loss 0.0
current train loss 2585594140.9960938
curren stop limit 0
2 epoch took 14.08s
For k = 8 running on epoch 32, with loss 2857586.0976867676 and lr 0.0005, and best loss of 2664985.541770935
user_emb_mean 0.5835089087486267 and user_emb_max 0.9780124425888062 and plan_emb_mean 0.30513250827789307 and plan_emb_max 0.6234263777732849
user_bias_mean -0.23200425505638123 and plan_bias_mean -0.8485925793647766 and last_grad_norm 1.1564478874206543 and d_loss 7.2270769539652235
current train loss 1810889478.861084
curren stop limit 6
2 epoch took 15.42s
For k = 12 running on epoch 12, with loss 3159996.1251277924 and lr 0.001, and best loss of 3159996.1251277924
user_emb_mean 0.6468802094459534 and user_emb_max 0.9718421101570129 and plan_emb_mean 0.4221312701702118 and plan_emb_max 0.6366583108901978
user_bias_mean -0.02987823076546192 and plan_bias_mean -0.4874849319458008 and last_grad_norm 1.5646779537200928 and d_loss 0.0
current train loss 2450124573.387451
curren stop limit 0
2 epoch took 13.87s
For k = 8 running on epoch 34, with loss 2884803.6537132263 and lr 0.0005, and best loss of 2664985.541770935
user_emb_mean 0.5799524784088135 and user_emb_max 0.9817777276039124 and plan_emb_mean 0.30463477969169617 and plan_emb_max 0.6239252686500549
user_bias_mean -0.24289561808109283 and plan_bias_mean -0.8575032353401184 and last_grad_norm 1.1467046737670898 and d_loss 8.248379156166747
current train loss 1797159202.178955
curren stop limit 7
2 epoch took 15.30s
For k = 12 running on epoch 14, with loss 2918440.5027275085 and lr 0.001, and best loss of 2918440.5027275085
user_emb_mean 0.6388745307922363 and user_emb_max 0.9714708924293518 and plan_emb_mean 0.3927183449268341 and plan_emb_max 0.6152082085609436
user_bias_mean -0.05294359475374222 and plan_bias_mean -0.5531507134437561 and last_grad_norm 1.485350251197815 and d_loss 0.0
current train loss 2332972043.7316895
curren stop limit 0
k = 8, converged at epoch 22, last epoch is 36, and best loss of the run is 2664985.54
Run took 505.022034406662s
2 epoch took 17.22s
For k = 12 running on epoch 16, with loss 2785925.3616600037 and lr 0.001, and best loss of 2785925.3616600037
user_emb_mean 0.6312177777290344 and user_emb_max 0.971100926399231 and plan_emb_mean 0.37504902482032776 and plan_emb_max 0.598474383354187
user_bias_mean -0.07593873143196106 and plan_bias_mean -0.6112478971481323 and last_grad_norm 1.4204320907592773 and d_loss 0.0
current train loss 2235054266.94458
curren stop limit 0
2 epoch took 18.29s
For k = 16 running on epoch 2, with loss 9318059.237350464 and lr 0.001, and best loss of 9318059.237350464
user_emb_mean 0.6936190724372864 and user_emb_max 0.9737028479576111 and plan_emb_mean 0.6450411081314087 and plan_emb_max 0.9013045430183411
user_bias_mean 0.08460590988397598 and plan_bias_mean -0.020302360877394676 and last_grad_norm 2.407259464263916 and d_loss 0.0
current train loss 3597083776.4819336
curren stop limit 0
2 epoch took 15.88s
For k = 12 running on epoch 18, with loss 2706957.3942012787 and lr 0.001, and best loss of 2706957.3942012787
user_emb_mean 0.6237459778785706 and user_emb_max 0.9707321524620056 and plan_emb_mean 0.36754900217056274 and plan_emb_max 0.5856011509895325
user_bias_mean -0.09881589561700821 and plan_bias_mean -0.662253737449646 and last_grad_norm 1.363916039466858 and d_loss 0.0
current train loss 2159168165.817871
curren stop limit 0
2 epoch took 19.40s
For k = 16 running on epoch 4, with loss 7344858.447689056 and lr 0.001, and best loss of 7344858.447689056
user_emb_mean 0.683380126953125 and user_emb_max 0.9733290076255798 and plan_emb_mean 0.5916968584060669 and plan_emb_max 0.8368125557899475
user_bias_mean 0.062138646841049194 and plan_bias_mean -0.1343533992767334 and last_grad_norm 2.1983211040496826 and d_loss 0.0
current train loss 3317248418.0615234
curren stop limit 0
2 epoch took 16.68s
For k = 12 running on epoch 20, with loss 2674968.765756607 and lr 0.001, and best loss of 2674968.765756607
user_emb_mean 0.6163076758384705 and user_emb_max 0.9703633785247803 and plan_emb_mean 0.36394548416137695 and plan_emb_max 0.5761398673057556
user_bias_mean -0.12155701965093613 and plan_bias_mean -0.7068394422531128 and last_grad_norm 1.33054780960083 and d_loss 0.0
current train loss 2100680366.9458008
curren stop limit 0
2 epoch took 18.48s
For k = 16 running on epoch 6, with loss 5891000.103336334 and lr 0.001, and best loss of 5891000.103336334
user_emb_mean 0.6735599040985107 and user_emb_max 0.9729564189910889 and plan_emb_mean 0.5427535176277161 and plan_emb_max 0.7793026566505432
user_bias_mean 0.039289288222789764 and plan_bias_mean -0.2374376803636551 and last_grad_norm 2.019986391067505 and d_loss 0.0
current train loss 3077771384.6972656
curren stop limit 0
2 epoch took 15.58s
For k = 12 running on epoch 22, with loss 2675353.8937950134 and lr 0.001, and best loss of 2674968.765756607
user_emb_mean 0.6088540554046631 and user_emb_max 0.9699947834014893 and plan_emb_mean 0.36140692234039307 and plan_emb_max 0.5692304968833923
user_bias_mean -0.14415216445922852 and plan_bias_mean -0.7456102967262268 and last_grad_norm 1.2942447662353516 and d_loss 0.014397477956997367
current train loss 2051574486.0095215
curren stop limit 1
2 epoch took 14.77s
For k = 12 running on epoch 24, with loss 2691000.963405609 and lr 0.001, and best loss of 2674968.765756607
user_emb_mean 0.60139000415802 and user_emb_max 0.9696261882781982 and plan_emb_mean 0.35953354835510254 and plan_emb_max 0.564279317855835
user_bias_mean -0.1665966957807541 and plan_bias_mean -0.7791542410850525 and last_grad_norm 1.2649005651474 and d_loss 0.5993414896740829
current train loss 2007933488.6694336
curren stop limit 2
2 epoch took 18.11s
For k = 16 running on epoch 8, with loss 4834937.803199768 and lr 0.001, and best loss of 4834937.803199768
user_emb_mean 0.6641967296600342 and user_emb_max 0.9725849032402039 and plan_emb_mean 0.49804699420928955 and plan_emb_max 0.7293316125869751
user_bias_mean 0.01623176597058773 and plan_bias_mean -0.3303743004798889 and last_grad_norm 1.869025468826294 and d_loss 0.0
current train loss 2872517690.1953125
curren stop limit 0
2 epoch took 15.33s
For k = 12 running on epoch 26, with loss 2716459.4488048553 and lr 0.001, and best loss of 2674968.765756607
user_emb_mean 0.5939252376556396 and user_emb_max 0.9692588448524475 and plan_emb_mean 0.35818666219711304 and plan_emb_max 0.5606750845909119
user_bias_mean -0.18890038132667542 and plan_bias_mean -0.8080917000770569 and last_grad_norm 1.2471057176589966 and d_loss 1.551071682757117
current train loss 1968565885.4821777
curren stop limit 3
2 epoch took 18.01s
For k = 16 running on epoch 10, with loss 4080477.9408988953 and lr 0.001, and best loss of 4080477.9408988953
user_emb_mean 0.6552886366844177 and user_emb_max 0.9722135663032532 and plan_emb_mean 0.4575643539428711 and plan_emb_max 0.6915109157562256
user_bias_mean -0.006941142026335001 and plan_bias_mean -0.4138895273208618 and last_grad_norm 1.747369408607483 and d_loss 0.0
current train loss 2696244055.7055664
curren stop limit 0
2 epoch took 15.35s
For k = 12 running on epoch 28, with loss 2748626.85641098 and lr 0.001, and best loss of 2674968.765756607
user_emb_mean 0.586468517780304 and user_emb_max 0.9688926339149475 and plan_emb_mean 0.35714519023895264 and plan_emb_max 0.558226466178894
user_bias_mean -0.21107657253742218 and plan_bias_mean -0.8330018520355225 and last_grad_norm 1.218907117843628 and d_loss 2.7536056344769766
current train loss 1932767454.9511719
curren stop limit 4
2 epoch took 18.41s
For k = 16 running on epoch 12, with loss 3562644.546087265 and lr 0.001, and best loss of 3562644.546087265
user_emb_mean 0.6468197107315063 and user_emb_max 0.9718422293663025 and plan_emb_mean 0.4256713390350342 and plan_emb_max 0.6600386500358582
user_bias_mean -0.03016272559762001 and plan_bias_mean -0.4886634349822998 and last_grad_norm 1.6397382020950317 and d_loss 0.0
current train loss 2544652653.92334
curren stop limit 0
2 epoch took 15.26s
For k = 12 running on epoch 30, with loss 2786561.77551651 and lr 0.0005, and best loss of 2674968.765756607
user_emb_mean 0.5790306925773621 and user_emb_max 0.9685265421867371 and plan_emb_mean 0.35633572936058044 and plan_emb_max 0.5564510822296143
user_bias_mean -0.2331356257200241 and plan_bias_mean -0.8544265627861023 and last_grad_norm 1.2033777236938477 and d_loss 4.171750010259996
current train loss 1899888383.5620117
curren stop limit 5
2 epoch took 18.01s
For k = 16 running on epoch 14, with loss 3223465.833328247 and lr 0.001, and best loss of 3223465.833328247
user_emb_mean 0.6387038230895996 and user_emb_max 0.9714710116386414 and plan_emb_mean 0.4096287786960602 and plan_emb_max 0.6342103481292725
user_bias_mean -0.05337204039096832 and plan_bias_mean -0.5552996397018433 and last_grad_norm 1.5681629180908203 and d_loss 0.0
current train loss 2420297370.476074
curren stop limit 0
2 epoch took 15.51s
For k = 12 running on epoch 32, with loss 2807400.6980571747 and lr 0.0005, and best loss of 2674968.765756607
user_emb_mean 0.5752440094947815 and user_emb_max 0.9683446288108826 and plan_emb_mean 0.35598328709602356 and plan_emb_max 0.5560494661331177
user_bias_mean -0.24410802125930786 and plan_bias_mean -0.8638343214988708 and last_grad_norm 1.2042770385742188 and d_loss 4.950784248245592
current train loss 1890045625.1428223
curren stop limit 6
2 epoch took 18.22s
For k = 16 running on epoch 16, with loss 3033459.9510860443 and lr 0.001, and best loss of 3033459.9510860443
user_emb_mean 0.6307289600372314 and user_emb_max 0.9711011052131653 and plan_emb_mean 0.40364137291908264 and plan_emb_max 0.6134428381919861
user_bias_mean -0.07651863992214203 and plan_bias_mean -0.6143613457679749 and last_grad_norm 1.4810316562652588 and d_loss 0.0
current train loss 2329000098.487549
curren stop limit 0
2 epoch took 15.66s
For k = 12 running on epoch 34, with loss 2829635.2490081787 and lr 0.0005, and best loss of 2674968.765756607
user_emb_mean 0.5714315176010132 and user_emb_max 0.9681627154350281 and plan_emb_mean 0.3556559681892395 and plan_emb_max 0.555634617805481
user_bias_mean -0.2550320029258728 and plan_bias_mean -0.8723871111869812 and last_grad_norm 1.1937015056610107 and d_loss 5.78199212011452
current train loss 1875108696.0107422
curren stop limit 7
2 epoch took 19.63s
For k = 16 running on epoch 18, with loss 2930052.775812149 and lr 0.001, and best loss of 2930052.775812149
user_emb_mean 0.6227633357048035 and user_emb_max 0.9707322716712952 and plan_emb_mean 0.39997485280036926 and plan_emb_max 0.5970053672790527
user_bias_mean -0.09953448176383972 and plan_bias_mean -0.6661843657493591 and last_grad_norm 1.4307633638381958 and d_loss 0.0
current train loss 2259860300.8459473
curren stop limit 0
k = 12, converged at epoch 22, last epoch is 36, and best loss of the run is 2674968.77
Run took 558.185741186142s
2 epoch took 17.97s
For k = 16 running on epoch 20, with loss 2862254.644765854 and lr 0.001, and best loss of 2862254.644765854
user_emb_mean 0.6147998571395874 and user_emb_max 0.9703635573387146 and plan_emb_mean 0.39721545577049255 and plan_emb_max 0.5843930840492249
user_bias_mean -0.12239030748605728 and plan_bias_mean -0.7112916111946106 and last_grad_norm 1.393160104751587 and d_loss 0.0
current train loss 2200538139.053955
curren stop limit 0
2 epoch took 17.66s
For k = 16 running on epoch 22, with loss 2819882.221632004 and lr 0.001, and best loss of 2819882.221632004
user_emb_mean 0.6068510413169861 and user_emb_max 0.9699949026107788 and plan_emb_mean 0.3951979875564575 and plan_emb_max 0.5772228240966797
user_bias_mean -0.145097017288208 and plan_bias_mean -0.7505077719688416 and last_grad_norm 1.3611081838607788 and d_loss 0.0
current train loss 2147623542.91626
curren stop limit 0
2 epoch took 17.95s
For k = 16 running on epoch 24, with loss 2798832.3600788116 and lr 0.001, and best loss of 2798832.3600788116
user_emb_mean 0.598927915096283 and user_emb_max 0.9696263074874878 and plan_emb_mean 0.3936944305896759 and plan_emb_max 0.5712722539901733
user_bias_mean -0.16766832768917084 and plan_bias_mean -0.7845787405967712 and last_grad_norm 1.3312036991119385 and d_loss 0.0
current train loss 2100047218.4057617
curren stop limit 0
2 epoch took 18.03s
For k = 16 running on epoch 26, with loss 2794032.518342972 and lr 0.001, and best loss of 2794032.518342972
user_emb_mean 0.5910418033599854 and user_emb_max 0.9692589640617371 and plan_emb_mean 0.39253315329551697 and plan_emb_max 0.5662277936935425
user_bias_mean -0.19011233747005463 and plan_bias_mean -0.8141055107116699 and last_grad_norm 1.298288106918335 and d_loss 0.0
current train loss 2056916411.5490723
curren stop limit 0
2 epoch took 18.17s
For k = 16 running on epoch 28, with loss 2802041.7670402527 and lr 0.001, and best loss of 2794032.518342972
user_emb_mean 0.5832042098045349 and user_emb_max 0.9688927531242371 and plan_emb_mean 0.3916260302066803 and plan_emb_max 0.5623002052307129
user_bias_mean -0.21243597567081451 and plan_bias_mean -0.8396412134170532 and last_grad_norm 1.2722917795181274 and d_loss 0.28665552904984964
current train loss 2017476238.0285645
curren stop limit 1
2 epoch took 17.95s
For k = 16 running on epoch 30, with loss 2820351.5215682983 and lr 0.001, and best loss of 2794032.518342972
user_emb_mean 0.5754263997077942 and user_emb_max 0.9685267210006714 and plan_emb_mean 0.3909090459346771 and plan_emb_max 0.5592318177223206
user_bias_mean -0.23464369773864746 and plan_bias_mean -0.8616679310798645 and last_grad_norm 1.2526546716690063 and d_loss 0.9419719724999936
current train loss 1981194670.1367188
curren stop limit 2
2 epoch took 18.53s
For k = 16 running on epoch 32, with loss 2847334.1917915344 and lr 0.001, and best loss of 2794032.518342972
user_emb_mean 0.5677182674407959 and user_emb_max 0.9681606888771057 and plan_emb_mean 0.39033544063568115 and plan_emb_max 0.556659460067749
user_bias_mean -0.25674206018447876 and plan_bias_mean -0.8806183338165283 and last_grad_norm 1.2315460443496704 and d_loss 1.907696961242731
current train loss 1947600334.6728516
curren stop limit 3
2 epoch took 17.87s
For k = 16 running on epoch 34, with loss 2880965.985179901 and lr 0.001, and best loss of 2794032.518342972
user_emb_mean 0.5600923299789429 and user_emb_max 0.9677947163581848 and plan_emb_mean 0.3898715674877167 and plan_emb_max 0.5546950697898865
user_bias_mean -0.2787344753742218 and plan_bias_mean -0.8968420028686523 and last_grad_norm 1.2158907651901245 and d_loss 3.1113978189661893
current train loss 1916397480.9985352
curren stop limit 4
2 epoch took 17.87s
For k = 16 running on epoch 36, with loss 2920264.1735076904 and lr 0.0005, and best loss of 2794032.518342972
user_emb_mean 0.5525625348091125 and user_emb_max 0.9674299359321594 and plan_emb_mean 0.38949230313301086 and plan_emb_max 0.5531196594238281
user_bias_mean -0.3006259799003601 and plan_bias_mean -0.9106796383857727 and last_grad_norm 1.1956993341445923 and d_loss 4.517902148096026
current train loss 1887233282.0471191
curren stop limit 5
2 epoch took 17.81s
For k = 16 running on epoch 38, with loss 2941531.313747406 and lr 0.0005, and best loss of 2794032.518342972
user_emb_mean 0.5487648844718933 and user_emb_max 0.9672481417655945 and plan_emb_mean 0.38932302594184875 and plan_emb_max 0.5525630712509155
user_bias_mean -0.3115203082561493 and plan_bias_mean -0.9166706800460815 and last_grad_norm 1.1990571022033691 and d_loss 5.27906509448608
current train loss 1881684204.029541
curren stop limit 6
2 epoch took 18.02s
For k = 16 running on epoch 40, with loss 2963929.8861637115 and lr 0.0005, and best loss of 2794032.518342972
user_emb_mean 0.5449767112731934 and user_emb_max 0.9670664072036743 and plan_emb_mean 0.3891621232032776 and plan_emb_max 0.5520440936088562
user_bias_mean -0.32237130403518677 and plan_bias_mean -0.9220455884933472 and last_grad_norm 1.1904679536819458 and d_loss 6.080722636739354
current train loss 1868116420.4638672
curren stop limit 7
k = 16, converged at epoch 28, last epoch is 42, and best loss of the run is 2794032.52
Run took 766.1933493614197s
shape: (9, 2)
┌─────┬───────────┐
│ k   ┆ best_loss │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 0   ┆ 2.4841e6  │
│ 1   ┆ 2.5544e6  │
│ 2   ┆ 2.5897e6  │
│ 3   ┆ 2.5934e6  │
│ 4   ┆ 2.5961e6  │
│ 6   ┆ 2.6522e6  │
│ 8   ┆ 2.6650e6  │
│ 12  ┆ 2.6750e6  │
│ 16  ┆ 2.7940e6  │
└─────┴───────────┘

Time taken 2047.91s
'''



