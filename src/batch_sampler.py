import torch
from torch.utils.data import Sampler, DataLoader
import numpy as np

class UniquePairBatchSampler(Sampler):
    def __init__(self, dataset, indices, batch_size):
        self.dataset = dataset
        self.indices = indices  # Use only the provided subset of indices
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        seen_pairs = set()  # To track unique (item_a_id, item_b_id) combinations in the batch

        for idx in self.indices:
            # Extract the pair information using the dataset's __getitem__
            _, _, item_a_id, item_b_id, _, _ = self.dataset[idx]

            # Create a tuple for the pair
            pair = (item_a_id, item_b_id)

            # Check if the pair is already in the set for the current batch
            if pair not in seen_pairs:
                # If the pair is unique, add it to the batch and track the pair
                batch.append(idx)
                seen_pairs.add(pair)

            # If the batch reaches the desired size, yield it
            if len(batch) == self.batch_size:
                yield batch
                # Reset batch and seen_pairs for the next batch
                batch = []
                seen_pairs = set()

        # Yield any remaining items in the batch if not empty
        if len(batch) > 0:
            yield batch

    def __len__(self):
        # Number of batches is the ceiling of total items in the subset divided by batch size
        return int(np.ceil(len(self.indices) / self.batch_size))

# Example usage
# Assuming `custom_dataset` is an instance of your custom dataset class
# batch_size = 32
# batch_sampler = UniquePairBatchSampler(custom_dataset, batch_size)
# data_loader = DataLoader(custom_dataset, batch_sampler=batch_sampler)
