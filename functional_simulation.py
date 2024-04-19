import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset
from itertools import chain
from contextualized.easy import ContextualizedCorrelationNetworks

# We are first setting up the simple linear machine learning problem with a known C, X, Y as well as the things to predict " the betas (sample specific models)"
n_samples = 50
n_observed = 3
n_context = 10
n_outcomes = 5

C = np.random.uniform(-1, 1, size=(n_samples, n_context))
X = np.random.uniform(0, 1, size=(n_samples, n_observed))
phi = np.random.uniform(-1, 1, size=(n_context, n_observed, n_outcomes))
beta = np.tensordot(C, phi, axes=1) + np.random.normal(0, 0.01, size=(n_samples, n_observed, n_outcomes))
Y = np.array([np.tensordot(X[i], beta[i], axes=1) for i in range(n_samples)])

# First we run this through a regular contextualized correlation model (just c and x) to get C' that becomes our second set of context to be used later

ccn = ContextualizedCorrelationNetworks(encoder_type='ngam', num_archetypes=16, n_bootstraps=1)
ccn.fit(C, X, max_epochs=5)

models = ccn.predict_networks(C, with_offsets=False, individual_preds=True)

observations = torch.from_numpy(X)
naive_context = torch.from_numpy(C)
print(len(naive_context))
synthetic_context = torch.from_numpy(models) # This is our synthetic context that we developed from the upstream model
#print(f"the size of the synthetic_context is = {synthetic_context.shape}")

final_raw_data = combine_tensor_datasets(naive_context, synthetic_context)


g_function_ids = assign_g_functions_interactively(final_raw_data)
processed_samples = apply_g_functions_to_samples(final_raw_data, g_function_ids)



first_sample = processed_samples[0]



prep_funcs, context_encoders = dynamic_prep_and_encoder_assignment(first_sample, k=4, output_size=100, hidden_layer_sizes= [64,64])
weighted_summation = WeightedSummation(num_subtypes = 2)
archetype_dictionary = ArchetypeDictionary(num_genes = 3, num_archetypes = 4)

optimizer = optim.Adam(
    params=list(chain(
        *(model.parameters() for model in prep_funcs),  # Assuming prep_funcs is a list of models
        *(model.parameters() for model in context_encoders),  # Assuming context_encoders is a list of models
        weighted_summation.parameters(),
        archetype_dictionary.parameters()
    )),
    lr=0.001
)

targets = torch.from_numpy(Y)


for epoch in range(20):
    total_loss = 0

    for sample, observation, target in zip(processed_samples, observations, targets):
        #for data in sample:
            #print(f"Data type before processing: {data.dtype}")
            #print(f"Shape of each affiliated dataset in a Sample: {data.shape}")

        optimizer.zero_grad()

        #print(f"Shape of target = {target.shape}")

        y_hat = forward_pass_regression(observation, sample, prep_funcs, context_encoders, weighted_summation, archetype_dictionary)

        

        #print(f"Shape of y_hat = {y_hat.shape}")
        #print(f"shape of  target  = {target.shape}")

        loss = F.mse_loss(y_hat, target)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
    if epoch % 1 == 0:
        print(f"Epoch {epoch + 1}: Loss = {total_loss / 50}")