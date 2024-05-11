# Use case with Data that has actual meaning
pip install contextualized-ml
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset
from itertools import chain
from contextualized.easy import ContextualizedCorrelationNetworks

# We are first setting up the simple linear machine learning problem with a known C, X, Y as well as the things to predict " the betas (sample specific models)"
n_samples =75
n_observed = 3 # Rename to Predictiors
n_context = 10
n_outcomes = 5


#Preliminary Factors (C ---> X)
A = torch.from_numpy(np.random.uniform(-1, 1, size=(n_context, n_observed)))
noise = torch.from_numpy(np.random.normal(0, 0.1, size=(n_samples, n_observed)))

#Training Data
C_train = torch.from_numpy(np.random.uniform(-1, 1, size=(n_samples, n_context)))
X_train = torch.mm(C_train, A) + noise
#X_train = torch.from_numpy(np.random.uniform(0, 1, size=(n_samples, n_observed)))
phi_train = torch.from_numpy(np.random.uniform(-1, 1, size=(n_context, n_observed, n_outcomes)))
beta_train = torch.from_numpy(np.tensordot(C_train, phi_train, axes=1) + np.random.normal(0, 0.01, size=(n_samples, n_observed, n_outcomes)))
Y_train = torch.from_numpy(np.array([np.tensordot(X_train[i], beta_train[i], axes=1) for i in range(n_samples)]))

#Validation Data
C_val =torch.from_numpy(np.random.uniform(-1, 1, size=(n_samples, n_context)))
X_val = torch.mm(C_val, A) + noise
#X_val = torch.from_numpy(np.random.uniform(0, 1, size=(n_samples, n_observed)))
phi_val = torch.from_numpy(np.random.uniform(-1, 1, size=(n_context, n_observed, n_outcomes)))
beta_val = torch.from_numpy(np.tensordot(C_val, phi_val, axes=1) + np.random.normal(0, 0.01, size=(n_samples, n_observed, n_outcomes)))
Y_val = torch.from_numpy(np.array([np.tensordot(X_val[i], beta_val[i], axes=1) for i in range(n_samples)]))

# Test Data
C_test = torch.from_numpy(np.random.uniform(-1, 1, size=(n_samples, n_context)))
X_test = torch.mm(C_test, A) + noise
#X_test = torch.from_numpy(np.random.uniform(0, 1, size=(n_samples, n_observed)))
phi_test = torch.from_numpy(np.random.uniform(-1, 1, size=(n_context, n_observed, n_outcomes)))
beta_test = torch.from_numpy(np.tensordot(C_test, phi_test, axes=1) + np.random.normal(0, 0.01, size=(n_samples, n_observed, n_outcomes)))
Y_test = torch.from_numpy(np.array([np.tensordot(X_test[i], beta_test[i], axes=1) for i in range(n_samples)]))


# Combining Training and Validation Data for Contextualized Regression Model
C_combined = torch.cat((C_train, C_val), dim=0)
X_combined = torch.cat((X_train, X_val), dim=0)
Y_combined = torch.cat((Y_train, Y_val), dim=0)



# First we run this through a regular contextualized correlation model (just c and x) to get C' that becomes our second set of context to be used later

#Training Data
ccn_train = ContextualizedCorrelationNetworks(encoder_type='ngam', num_archetypes=16, n_bootstraps=1)
ccn_train.fit(C_train, X_train, max_epochs=5)


#Validation Data
ccn_val = ContextualizedCorrelationNetworks(encoder_type='ngam', num_archetypes=16, n_bootstraps=1)
ccn_val.fit(C_val, X_val, max_epochs=5)

# Test Data
ccn_test = ContextualizedCorrelationNetworks(encoder_type='ngam', num_archetypes=16, n_bootstraps=1)
ccn_test.fit(C_test, X_test, max_epochs=5)

#Running Dataset through upstream model (Transfer Learning in Contextualized Framework)
models_train = ccn_train.predict_networks(C_train, with_offsets=False, individual_preds=False)
models_val = ccn_val.predict_networks(C_val, with_offsets=False, individual_preds=False)
models_test = ccn_test.predict_networks(C_test, with_offsets=False, individual_preds=False)

# Process With Training Data and Validation Data

# Training Data
observations_train = X_train
naive_context_train = C_train
synthetic_context_train = torch.from_numpy(models_train)


final_raw_data_train = combine_tensor_datasets(naive_context_train, synthetic_context_train)
g_function_ids_train = assign_g_functions_interactively(final_raw_data_train)
processed_samples_train = apply_g_functions_to_samples(final_raw_data_train, g_function_ids_train)
first_sample_train = processed_samples_train[0]
targets_train = Y_train


# Validation Data
observations_val = X_val
naive_context_val = C_val
synthetic_context_val = torch.from_numpy(models_val)  # Assuming models_val is analogous to models_train

final_raw_data_val = combine_tensor_datasets(naive_context_val, synthetic_context_val)
g_function_ids_val = assign_g_functions_interactively(final_raw_data_val)  # Assumes interactive can be rerun for validation
processed_samples_val = apply_g_functions_to_samples(final_raw_data_val, g_function_ids_val)
targets_val = Y_val


# Optimization Steps
prep_funcs, context_encoders = dynamic_prep_and_encoder_assignment(first_sample_train, k=4, output_size= 15, hidden_layer_sizes= [4,4])
weighted_summation = WeightedSummation(num_subtypes = 2)
archetype_dictionary = ArchetypeDictionary(observation_count=3, output_count=5, num_archetypes=4)

optimizer = optim.Adam(
    params=list(chain(
        *(model.parameters() for model in prep_funcs),  # Assuming prep_funcs is a list of models
        *(model.parameters() for model in context_encoders),  # Assuming context_encoders is a list of models
        weighted_summation.parameters(),
        archetype_dictionary.parameters()
    )),
    lr=0.0001
)

best_model_weights = {}
lowest_val_loss = float('inf')

for epoch in range(200):
    total_loss_train = 0

    # Training phase
    for sample, observation, target in zip(processed_samples_train, observations_train, targets_train):
        optimizer.zero_grad()
        y_hat = forward_pass_regression(observation, sample, prep_funcs, context_encoders, weighted_summation, archetype_dictionary)
        loss = F.mse_loss(y_hat, target)
        loss.backward()
        optimizer.step()
        total_loss_train += loss.item()

    # Validation phase
    with torch.no_grad():
        total_loss_val = 0
        for sample, observation, target in zip(processed_samples_val, observations_val, targets_val):
            y_hat = forward_pass_regression(observation, sample, prep_funcs, context_encoders, weighted_summation, archetype_dictionary)
            loss = F.mse_loss(y_hat, target)
            total_loss_val += loss.item()

    if total_loss_val < lowest_val_loss:
        lowest_val_loss = total_loss_val
        best_model_weights = {
            'prep_funcs': [func.state_dict() for func in prep_funcs],
            'context_encoders': [encoder.state_dict() for encoder in context_encoders],
            'weighted_summation': weighted_summation.state_dict(),
            'archetype_dictionary': archetype_dictionary.state_dict(),
        }
    else:
        print(f"Stopping early at epoch {epoch + 1} due to increase in validation loss.")
        break

    print(f"Epoch {epoch + 1}: Validation Loss = {total_loss_val / len(targets_val)}")
    

# Basleine Contextualized ML Regression Model 
from contextualized.easy import ContextualizedRegressor
model_regression = ContextualizedRegressor()
model_regression.fit(C_combined, X_combined, Y_combined, max_epochs=-1, learning_rate=1e-3,val_split=0.5, n_bootstraps=1)

y_hats = model_regression.predict(C_test, X_test, individual_preds=False)

total_loss_val = 0
y_hats = torch.from_numpy(y_hats).double()
for y_hat, target in zip(y_hats, Y_test):
    #print(f"size of y_hat is = {y_hat.shape}")
    #print(f"size of target is = {target.dtype}")
    loss = F.mse_loss(y_hat, target)
    total_loss_val += loss.item()
print(f"The average loss is = {total_loss_val/150}")


# UNIT TESTING OF CTL 

observations_test = X_test
naive_context_test = C_test
synthetic_context_test = torch.from_numpy(models_test)  # Assuming models_val is analogous to models_train

final_raw_data_test = combine_tensor_datasets(naive_context_test, synthetic_context_test)
g_function_ids_test = assign_g_functions_interactively(final_raw_data_test)  # Assumes interactive can be rerun for validation
processed_samples_test = apply_g_functions_to_samples(final_raw_data_test, g_function_ids_test)
targets_test = Y_test

y_hats_test = predict(observations_test, processed_samples_test, best_model_weights)

total_loss_val = 0
#y_hats_test = torch.from_numpy(y_hats_test).double()
for y_hat, target in zip(y_hats_test, targets_test):
    #print(f"size of y_hat is = {y_hat.shape}")
    #print(f"size of target is = {target.dtype}")
    loss = F.mse_loss(y_hat, target)
    total_loss_val += loss.item()
print(f"The average loss is = {total_loss_val/150}")
