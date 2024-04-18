# Sample Use Case Functional 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset
from itertools import chain


# Assuming all class definitions (PreparationFunction, LinearPreparationFunction,
# MLPPreparationFunction, ContextEncoder, SubModel, WeightedSummation, ArchetypeDictionary)
# and functions (assign_g_functions, g1_identity, g2_imagenet, g3_text, generate_synthetic_upstream_data)
# are defined as previously stated

# 1. Generate synthetic upstream data
num_samples = 3
feature_size = 10
num_archetypes = 2
synthetic_upstream_data = generate_synthetic_upstream_data(num_samples, feature_size)
#print(f"synthetic_upstream_data = {len(synthetic_upstream_data)}")

# 2. Simulate raw data for samples
samples_raw_data = [
    [torch.randn(1, 10), torch.randn(1, 15), torch.randn(1, 10)] for _ in range(num_samples)
]


final_raw_data = combine_tensor_datasets(samples_raw_data, synthetic_upstream_data)
print(f"samples in final raw data = {len(final_raw_data)}")

#g_function_ids = assign_g_functions_interactively(final_raw_data)
g_function_ids = [1,1,1,0]



# 4. Apply G functions, incorporating synthetic upstream data
processed_samples = apply_g_functions_to_samples(final_raw_data, g_function_ids)
print(f"size of processed_samples = {len(processed_samples)}")

first_sample = processed_samples[0]


#for sample in processed_samples:
    #print(f"Sample length: {len(sample)}")
    #for data in sample:
        #print(f"Data shape: {data.shape}")

# 5. Dynamic assignment of preparation functions and context encoders
# Assuming all data sets within a sample type require similar processing, we can generalize the assignment
prep_funcs, context_encoders = dynamic_prep_and_encoder_assignment(first_sample, k=2, output_size=100, hidden_layer_sizes= [64,64])
print(f"prep_funcs={len(prep_funcs)}")
print(f"context_encoders={len(context_encoders)}")

# 6. Create Weighted Summation and Archetype Dictionary instances
weighted_summation = WeightedSummation(num_subtypes=4)
archetype_dictionary = ArchetypeDictionary(num_genes = 5, num_archetypes = 2)
#print(weighted_summation)
#print(f"archetype_dictionary={archetype_dictionary}")


# 7. Initialize the optimizer
optimizer = optim.Adam(
    params=list(chain(
        *(model.parameters() for model in prep_funcs),  # Assuming prep_funcs is a list of models
        *(model.parameters() for model in context_encoders),  # Assuming context_encoders is a list of models
        weighted_summation.parameters(),
        archetype_dictionary.parameters()
    )),
    lr=0.001
)


# 8. Define a forward pass function (not repeating the definition here, assuming it's defined correctly)

# 9. Training loop with optimization
targets = [torch.randn(1, 5, 5) for _ in range(num_samples)]  # Random targets for each sample (SAMPLE SPECIFIC MODEL)

for epoch in range(15):  # Example: 100 epochs
    total_loss = 0

    # Loop through each sample and its corresponding target
    for sample, target in zip(processed_samples, targets):
        optimizer.zero_grad()  # Clear gradients before each operation
        
        # Compute the model output for the current sample
        sample_specific_model = forward_pass(sample, prep_funcs, context_encoders, weighted_summation, archetype_dictionary)
        print("sample ran through")
        
        # Compute loss
        loss = F.mse_loss(sample_specific_model, target)
        
        # Perform backpropagation
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()

    if epoch % 3 == 0:
        print(f"Epoch {epoch + 1}: Loss = {total_loss / num_samples}")