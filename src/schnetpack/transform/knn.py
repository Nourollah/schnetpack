import torch

from .base import Transform
import schnetpack.properties as structure
import itertools


class KNNRepresentation(Transform):
	"""_summary_

	Args:
		Transform (_type_): _description_
	"""

	def __init__(self,
	             k: int,
	             threshold: float,
	             embedding_reserved: int = 32):
		"""
		k (int): The number of nearest neighbors to find.
		threshold (float): The maximum distance to consider for neighboring nodes.
		"""
		super().__init__()
		self.k = k
		self.threshold = threshold
		self.embedding_reserved = embedding_reserved

	def forward(self, inputs):
		return self.feature_maker(inputs)

	def _find_k_nearest_neighbors(self, positions):
		"""
		Find the K nearest neighbors for each node in 3D space.

		Args:
		positions (torch.Tensor): A tensor of shape (num_nodes, 3) containing the x, y, z coordinates of each node.
		k (int): The number of nearest neighbors to find.
		threshold (float): The maximum distance to consider for neighboring nodes.

		Returns:
		torch.Tensor: A tensor of shape (num_nodes, k) containing the indices of the K nearest neighbors for each node.
				Padding value of -1 is used if fewer than K neighbors are found within the threshold.
		torch.Tensor: A tensor of shape (num_nodes, k) containing the distances to the K nearest neighbors.
				Padding value of inf is used if fewer than K neighbors are found within the threshold.
		"""
		num_nodes = positions.shape[0]

		# Compute pairwise distances between all nodes
		distances = torch.cdist(positions, positions)

		# Set diagonal elements to inf to exclude self-connections
		distances.fill_diagonal_(float('inf'))

		# Apply distance threshold
		distances[distances > self.threshold] = float('inf')

		# Find the K nearest neighbors
		k = min(self.k, num_nodes - 1)  # Ensure k is not larger than num_nodes - 1
		distances, indices = torch.topk(distances, k, largest=False, sorted=True)

		# Create masks for valid neighbors (distance < threshold)
		valid_neighbors = distances < float('inf')

		# Pad indices and distances where there are fewer than k valid neighbors
		indices = torch.where(valid_neighbors, indices, torch.tensor(-1, device=indices.device))
		distances = torch.where(valid_neighbors, distances, torch.tensor(float('inf'), device=distances.device))

		return indices, distances

	def feature_maker(self, inputs):
		inputs[structure.knn], _ = self._find_k_nearest_neighbors(inputs[structure.R])
		# Computing the Size of the embedding of z_i and z_j plus the distance between these tow and whatever considered as a feature
		third_dim_size: int = 1 + (2 * self.embedding_reserved)
		# Creating a 3D tensor to keep all features of atoms
		knn_representation: torch.Tensor = torch.zeros(inputs[structure.knn].shape[0], self.k, third_dim_size)
		for the_atom, the_neighbor_atom in itertools.product(range(inputs[structure.knn].shape[0]),
		                                                     range(inputs[structure.knn].shape[1])):
			neighbor_number: int = inputs[structure.knn][the_atom, the_neighbor_atom]
			# Skip when there is no neighbor for the atom
			if not neighbor_number >= 0 and neighbor_number <= self.k:
				continue
			neighbor_position: torch.Tensor = inputs[structure.R][neighbor_number, :]
			the_atom_position: torch.Tensor = inputs[structure.R][the_atom, :]
			the_distance: torch.Tensor = torch.sqrt(torch.sum((the_atom_position - neighbor_position) ** 2))
			# the_atom_representation = self._embedding(inputs[structure.Z][the_atom])[:, None]
			# the_neighbor_representation = self._embedding(inputs[structure.Z][neighbor_number])[:, None]
			knn_representation[the_atom, the_neighbor_atom, 0:1] = the_distance
			knn_representation[the_atom, the_neighbor_atom, 1:2] = inputs[structure.Z][the_atom]
			knn_representation[the_atom, the_neighbor_atom, 2:3] = inputs[structure.Z][neighbor_number]
			# knn_representation[the_atom, the_neighbor_atom, 1:embedding_size + 1] = torch.squeeze(
			# 	the_atom_representation)
			# knn_representation[the_atom, the_neighbor_atom,
			# embedding_size + 1:2 * embedding_size + 1] = torch.squeeze(the_neighbor_representation)
		inputs[structure.knn] = knn_representation
		# print("********************")
		# print(f"{inputs[structure.knn].size()}")
		# print("********************")

		return inputs
