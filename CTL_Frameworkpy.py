import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Critical Functions


def generate_synthetic_upstream_data(num_samples, feature_size):
    """
    Generate synthetic data to simulate the output from an upstream model.

    :param num_samples: Number of samples to generate data for.
    :param feature_size: Size of the feature vector for each sample.
    :return: A tensor representing the synthetic upstream model data for all samples.
    """
    return torch.randn(num_samples, feature_size)


def module_to_tensor_list(module):
    """
    Extracts all tensors from a torch.nn.Module and returns them as a list of lists (each tensor wrapped in a list).

    Args:
    - module (torch.nn.Module): The module from which to extract tensors.

    Returns:
    - List[List[torch.Tensor]]: A list where each element is a list containing a single tensor.
    """
    # Extract parameters (which are tensors) and convert the iterator to a list
    tensor_list = list(module.parameters())

    # Wrap each tensor in a list to conform to the expected input of combine_tensor_datasets
    wrapped_tensor_list = [[tensor] for tensor in tensor_list]

    return wrapped_tensor_list


def combine_tensor_datasets(*datasets):
    """
    Combine multiple datasets where each dataset is a list of tensor samples.
    The first dataset contains samples with multiple affiliated datasets (tensors),
    and subsequent datasets contain samples with a single or multiple affiliated dataset tensors.
    The result is samples each with an extended number of affiliated datasets (tensors).

    This function supports combining any number of datasets greater than or equal to 1.

    Args:
    - datasets: A sequence of datasets, where each dataset is a list of samples. Each sample can be
      a single tensor or a list of affiliated dataset tensors.

    Returns:
    - A new combined dataset with the same structure as the first dataset.
    """
    if not datasets:
        raise ValueError("At least one dataset must be provided.")

    # Check if all datasets have the same number of samples
    num_samples = len(datasets[0])
    if not all(len(dataset) == num_samples for dataset in datasets):
        raise ValueError("All datasets must contain the same number of samples.")

    combined_dataset = []
    for idx in range(num_samples):
        combined_sample = []
        # Iterate through each dataset and collect the sample at the current index
        for dataset in datasets:
            sample = dataset[idx]
            # If the sample is not a list (a single tensor), wrap it in a list
            if not isinstance(sample, list):
                sample = [sample]
            combined_sample.extend(sample)
        combined_dataset.append(combined_sample)

    return combined_dataset





# G Function Assignment (Cataloug) Function

def assign_g_functions_interactively(samples):
    """
    Prompt the user to assign a G function for each affiliated dataset within the first sample.
    These assignments are then applied to all samples in the dataset.

    Args:
    - samples: A dataset consisting of multiple samples, each with one or more affiliated datasets.

    Returns:
    - A list of G function assignments for each affiliated dataset, applied to all samples.
    """
    g_function_descriptions = {
        0: "Upstream Model (G0)",
        1: "Identity Function (G1)",
        2: "ImageNet Classifier (G2)"
    }

    # Display G function options
    print("Available G Functions:")
    for g_id, description in g_function_descriptions.items():
        print(f"{g_id}: {description}")

    g_assignments = []
    # Only prompt for the first sample
    if samples:
        first_sample = samples[0]
        for i, _ in enumerate(first_sample):
            while True:
                try:
                    # Prompt user for G function assignment for each affiliated dataset in the first sample
                    choice = int(input(f"Choose a G function for dataset {i+1} of the first sample: "))
                    if choice in g_function_descriptions:
                        g_assignments.append(choice)
                        break
                    else:
                        print("Invalid choice. Please choose from the available G functions.")
                except ValueError:
                    print("Please enter a valid integer.")

    # Apply these assignments to all samples
    return g_assignments



# Now `g_function_assignments` contains the G function IDs chosen by the user for the first sample,
# and these IDs are to be applied to all samples in the same order.


# Applier for G Functions

# Load a pretrained ResNet model
imagenet_model = models.resnet18(pretrained=True)

# Extract the first few layers of the model
# For ResNet, this usually includes: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
initial_layers = torch.nn.Sequential(
    list(imagenet_model.children())[0],  # Conv2d
    list(imagenet_model.children())[1],  # BatchNorm2d
    list(imagenet_model.children())[2],  # ReLU
    list(imagenet_model.children())[3]   # MaxPool2d
)

def apply_g_function(data, g_id):
    """
    Apply the specified G function to the data.
    For G function 2, apply the initial layers of the ImageNet model.
    Args:
    - data: The input data (tensor).
    - g_id: The G function identifier (0, 1, or 2).
    Returns:
    - The transformed data after applying the G function.
    """
    if g_id == 2:
        # Apply the initial layers of the ImageNet model
        data = initial_layers(data)  # Assuming data is a 4D tensor (batch_size, channels, height, width)
    # For g_id 0 and 1, data is returned as is
    return data

def apply_g_functions_to_samples(samples, g_function_assignments):
    """
    Apply the G functions to each dataset within all samples based on the provided assignments.
    Args:
    - samples: A list of samples, each sample containing multiple datasets.
    - g_function_assignments: A list of G function IDs to apply to each dataset within the samples.
    Returns:
    - A new list of samples with each dataset processed by its corresponding G function.
    """
    processed_samples = []
    for sample in samples:
        processed_sample = [apply_g_function(dataset, g_id) for dataset, g_id in zip(sample, g_function_assignments)]
        processed_samples.append(processed_sample)
    return processed_samples

# Linear Preperation/Context Encoder Assignment Function

class PreparationFunction(nn.Module):
    def __init__(self, input_shape, output_size):
        super().__init__()
        self.input_shape = input_shape
        self.output_size = output_size

    def forward(self, x):
        raise NotImplementedError("This method should be implemented by subclasses")



class LinearPreparationFunction(PreparationFunction):
    def __init__(self, input_shape, output_size):
        super().__init__(input_shape, output_size)
        self.linear = nn.Linear(int(input_shape), int(output_size))

    def forward(self, x):
        return self.linear(x.view(-1, self.input_shape))  # Flatten the input if not already 1D

class MLPPreparationFunction(PreparationFunction):
    def __init__(self, input_shape, output_size, hidden_layer_sizes=[64, 64]):
        super().__init__(input_shape, output_size)
        layers = []
        last_size = input_shape

        for size in hidden_layer_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size

        layers.append(nn.Linear(last_size, output_size))
        self.layers = nn.Sequential(*layers)  # Use Sequential for simplicity

    def forward(self, x):
        # Handle single sample dataset by flattening correctly
        if x.dim() == 2:  # Assuming the input is 3x3 without a batch dimension
            x = x.view(-1)  # Flatten to a vector
            x = x.unsqueeze(0)  # Add a batch dimension to mimic a single sample batch
        elif x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten maintaining the batch size
        return self.layers(x)


# Context Encoders

class SubModel(nn.Module):
    def __init__(self, hidden_size):
        super(SubModel, self).__init__()
        # Define a simple feed-forward network for a single feature
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),  # Single feature as input
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output a single value
        )

    def forward(self, x):
        # x is expected to be a single feature value
        return self.network(x)

class ContextEncoder(nn.Module):
    def __init__(self, num_features, hidden_size, output_size):
        super(ContextEncoder, self).__init__()
        self.num_features = num_features
        self.sub_models = nn.ModuleList([SubModel(hidden_size) for _ in range(num_features)])
        # Linear layer to project the sum of sub-model outputs to the archetype space of dimension k
        self.output_layer = nn.Linear(num_features, output_size)

    def forward(self, refined_context):
        # refined_context is expected to be a vector with 'num_features' elements
        # Apply each sub-model to its corresponding feature
        sub_outputs = torch.cat([sub_model(refined_context[:, i:i+1]) for i, sub_model in enumerate(self.sub_models)], dim=1)
        # Sum the outputs of the sub-models and project to k-dimensional space
        output = self.output_layer(sub_outputs)
        return output

# Linear Preperation/Context Encoder Assignment Function

def dynamic_prep_and_encoder_assignment(first_sample, k=5, output_size=100, hidden_layer_sizes=[64, 64]):
    """
    Initializes preparation functions and context encoders based on the first sample's datasets.
    These models are shared across all samples.

    Args:
    - first_sample: A list of tensors representing the first sample's datasets to determine the
                    model initialization.
    - k: Number of archetypes, output size for the context encoders.
    - output_size: Desired output size for the preparation functions.
    - hidden_layer_sizes: Hidden layer sizes for MLP preparation functions.

    Returns:
    - A tuple containing two lists: initialized preparation functions and context encoders.
    """
    shared_prep_funcs = []
    shared_context_encoders = []

    for data in first_sample:
        if data.dim() > 1:  # Multidimensional data
            # Flatten all dimensions to create a single long vector
            input_shape = data.numel()  # Number of elements in tensor
            prep_func = MLPPreparationFunction(input_shape=input_shape, output_size=output_size, hidden_layer_sizes=hidden_layer_sizes)
        else:  # Single-dimensional data
            input_shape = data.shape[-1]
            prep_func = LinearPreparationFunction(input_shape=input_shape, output_size=output_size)

        shared_prep_funcs.append(prep_func)

        # Initialize a shared context encoder for each preparation function's output
        context_encoder = ContextEncoder(num_features=output_size, hidden_size=hidden_layer_sizes[-1], output_size=k)
        shared_context_encoders.append(context_encoder)

    return shared_prep_funcs, shared_context_encoders





# Summation of Subtypes --> Super Subtype

class WeightedSummation(nn.Module):
    def __init__(self, num_subtypes):
        super(WeightedSummation, self).__init__()
        # Initialize the weights for each subtype as learnable parameters
        self.weights = nn.Parameter(torch.ones(num_subtypes, 1))  # Shape (num_subtypes, 1) for proper broadcasting

    def forward(self, subtypes):
        # Ensure subtypes is of shape (batch_size, num_subtypes, k)

        # Weights shape is (num_subtypes, 1) to broadcast correctly over the k dimension
        weighted_subtypes = subtypes * self.weights.unsqueeze(0)  # Adding batch dimension for broadcasting
        super_subtype = torch.sum(weighted_subtypes, dim=1)  # Sum over num_subtypes dimension
        return super_subtype


# Creation of Archetypes and Archetype Dictionary

class ArchetypeDictionary(nn.Module):
    def __init__(self, observation_count, output_count, num_archetypes):
        """
        Initializes the archetype dictionary.

        :param num_genes: The number of genes, m, which defines the shape of each archetype matrix.
        :param num_archetypes: The number of archetypes, k, in the dictionary.
        """
        super(ArchetypeDictionary, self).__init__()
        self.observation_count = observation_count
        self.output_count = output_count
        self.num_archetypes  = num_archetypes
        # Initialize the archetype dictionary as a learnable parameter
        # Each archetype is an m x m matrix, and there are k such matrices.
        self.archetypes = nn.Parameter(torch.randn(num_archetypes, observation_count, output_count))

    def forward(self, x):
        """
        Placeholder forward pass. In actual usage, the dot product with super subtype
        or other operations might be implemented here as needed.
        """
        return x

# Summation + Sigmoid Function for Super Subtype Dot Prod with Archetype Dictionary

def archetype_weighting(super_subtype, archetypes):
    """
    Applies sigmoid to the super subtype, then performs a weighted sum of the archetypes.
    """
    # Sigmoid to ensure values are between 0 and 1
    weights = F.sigmoid(super_subtype)  # Expected shape: [batch_size, num_archetypes]

    # Expand weights for proper broadcasting
    # After unsqueezing, expected weights shape: [batch_size, num_archetypes, 1, 1]
    weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)

    # Weighted sum of archetypes, broadcasting across batch and archetype dimensions
    # Expected output shape: [batch_size, num_genes, num_genes]
    weighted_archetypes = torch.sum(weights_expanded * archetypes.unsqueeze(0), dim=1)

    return weighted_archetypes


#Complete Forward Pass of the Whole Model

def forward_pass(sample, prep_funcs, context_encoders, weighted_summation, archetype_dictionary):
    # Process the sample
    refined_contexts = [prep_func(data.view(-1).unsqueeze(0) if data.dim() > 1 else data.unsqueeze(0)) for prep_func, data in zip(prep_funcs, sample)]

    encoded_contexts = [encoder(context) for encoder, context in zip(context_encoders, refined_contexts)]



    subtypes_tensor = torch.stack(encoded_contexts, dim=1)
    print(f"Subtypes_Tensor = {subtypes_tensor}")

    # Calculate super subtype for the current sample
    super_subtype = weighted_summation(subtypes_tensor)
    print(f"Super_Subtype = {super_subtype}")

    # Compute the sample-specific model using the current super subtype
    sample_specific_model = archetype_weighting(super_subtype, archetype_dictionary.archetypes)

    return sample_specific_model


def forward_pass_regression(observation, sample, prep_funcs, context_encoders, weighted_summation, archetype_dictionary):
    # Process the sample
    refined_contexts = [
    prep_func(data.view(-1).unsqueeze(0).float() if data.dim() > 1 else data.unsqueeze(0).float())
    for prep_func, data in zip(prep_funcs, sample)
    ]


    encoded_contexts = [encoder(context) for encoder, context in zip(context_encoders, refined_contexts)]


    subtypes_tensor = torch.stack(encoded_contexts, dim=1)



    # Calculate super subtype for the current sample
    super_subtype = weighted_summation(subtypes_tensor)



    # Compute the sample-specific model using the current super subtype
    sample_specific_model = archetype_weighting(super_subtype, archetype_dictionary.archetypes)

    #print(f"Shape of sample_specific_model is = {sample_specific_model.shape}")
    #print(f"Shape of Observation is = {observation.shape}")

    sample_specific_model = sample_specific_model.double()
    observation = observation.double()

    transposed_model = sample_specific_model.transpose(0, 2).squeeze(2)

    y_hat = torch.matmul(transposed_model, observation.T)

    return y_hat


# R^2 Calculation to give a bettter idea on how the model is performing

def compute_r2_score(predictions, targets):
    # Ensure predictions and targets are flattened for this calculation
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    target_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - target_mean) ** 2)
    ss_res = torch.sum((targets - predictions) ** 2)

    # Adding a small epsilon to ss_tot to avoid division by zero
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2

def predict(observation_test, processed_samples_test, best_model_weights):
    # Load the weights into models
    for idx, func in enumerate(prep_funcs):
        func.load_state_dict(best_model_weights['prep_funcs'][idx])
    for idx, encoder in enumerate(context_encoders):
        encoder.load_state_dict(best_model_weights['context_encoders'][idx])
    weighted_summation.load_state_dict(best_model_weights['weighted_summation'])
    archetype_dictionary.load_state_dict(best_model_weights['archetype_dictionary'])

    # Process each sample in the test dataset and predict
    predictions = []
    for observation, sample in zip(observation_test, processed_samples_test):
        y_hat = forward_pass_regression(observation, sample, prep_funcs, context_encoders, weighted_summation, archetype_dictionary)
        predictions.append(y_hat)

    return torch.stack(predictions)  # Stack predictions to form a batch