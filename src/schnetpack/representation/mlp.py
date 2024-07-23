import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.properties as properties
import schnetpack.nn as snn
from schnetpack.nn.embedding import NuclearEmbedding, ElectronicEmbedding
from rich.pretty import pprint

__all__ = ["MLP"]


class MLP(nn.Module):
	"""PaiNN - polarizable interaction neural network

	References:

	.. [#painn1] Schütt, Unke, Gastegger:
	   Equivariant message passing for the prediction of tensorial properties and molecular spectra.
	   ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

	"""

	def __init__(
			self,
			n_atom_basis: int,
			n_interactions: int,
			radial_basis: nn.Module,
			cutoff_fn: typing.Optional[typing.Callable] = None,
			activation: typing.Optional[typing.Callable] = F.silu,
			max_z: int = 101,
			shared_interactions: bool = False,
			shared_filters: bool = False,
			epsilon: float = 1e-8,
			activate_charge_spin_embedding: bool = False,
			embedding: typing.Union[typing.Callable, nn.Module] = None,
			distance_terms: int = 1,
	):
		"""
		Args:
			n_atom_basis: number of features to describe atomic environments.
				This determines the size of each embedding vector; i.e. embeddings_dim.
			n_interactions: number of interaction blocks.
			radial_basis: layer for expanding interatomic distances in a basis set
			cutoff_fn: cutoff function
			max_z: maximal nuclear charge
			activation: activation function
			shared_interactions: if True, share the weights across
				interaction blocks.
			shared_interactions: if True, share the weights across
				filter-generating networks.
			epsilon: stability constant added in norm to prevent numerical instabilities
			activate_charge_spin_embedding: if True, charge and spin embeddings are added
				to nuclear embeddings taken from SpookyNet Implementation
			embedding: custom nuclear embedding
		"""
		super(MLP, self).__init__()

		self.n_atom_basis = n_atom_basis
		self.n_interactions = n_interactions
		self.cutoff_fn = cutoff_fn
		self.cutoff = cutoff_fn.cutoff
		self.radial_basis = radial_basis
		self.activate_charge_spin_embedding = activate_charge_spin_embedding
		self.distance_terms = distance_terms

		# initialize nuclear embedding
		self.embedding = embedding
		if self.embedding is None:
			self.embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)

		# initialize spin and charge embeddings
		if self.activate_charge_spin_embedding:
			self.charge_embedding = ElectronicEmbedding(
				self.n_atom_basis,
				num_residual=1,
				activation=activation,
				is_charged=True)
			self.spin_embedding = ElectronicEmbedding(
				self.n_atom_basis,
				num_residual=1,
				activation=activation,
				is_charged=False)

		# initialize filter layers
		self.share_filters = shared_filters
		# if shared_filters:
		# 	self.filter_net = snn.Dense(
		# 		self.radial_basis.n_rbf, 3 * n_atom_basis, activation=None
		# 	)
		# else:
		# 	self.filter_net = snn.Dense(
		# 		self.radial_basis.n_rbf,
		# 		self.n_interactions * n_atom_basis * 3,
		# 		activation=None,
		# 	)
		self.feature_extractor_scalar = snn.Dense(n_atom_basis + distance_terms,
		                                          n_atom_basis,
		                                          activation=activation)
		self.perceptron_scalar = snn.replicate_module(
			lambda: snn.Dense(
				in_features=n_atom_basis,
				out_features=n_atom_basis,
				activation=activation
			),
			3,  # @TODO: convert to parameter later
			share_params=False,  # @TODO: convert to parameter later
		)

		self.feature_extractor_vector = snn.Dense(n_atom_basis + distance_terms * 3,
		                                          n_atom_basis,
		                                          activation=activation)
		self.perceptron_vector = snn.replicate_module(
			lambda: snn.Dense(
				in_features=n_atom_basis,
				out_features=n_atom_basis,
				activation=activation
			),
			3,  # @TODO: convert to parameter later
			share_params=False,  # @TODO: convert to parameter later
		)

	def forward(self, inputs: typing.Dict[str, torch.Tensor]):
		"""
		Compute atomic representations/embeddings.

		Args:
			inputs: SchNetPack dictionary of input tensors.

		Returns:
			torch.Tensor: atom-wise representation.
			list of torch.Tensor: intermediate atom-wise representations, if
			return_intermediate=True was used.
		"""
		# get tensors from input dictionary
		atomic_numbers = inputs[properties.Z]
		r_ij = inputs[properties.Rij]
		idx_i = inputs[properties.idx_i]
		idx_j = inputs[properties.idx_j]
		n_atoms = atomic_numbers.shape[0]
		pprint(f"atomic numbers: {atomic_numbers.size()}")
		pprint(f"inputs: {inputs}")
		# compute atom and pair features
		# This line of code is applying the Euclidean norm function to the tensor r_ij and 'd' indicate distance here (The biggest Value could be whatever)
		d_ij = torch.norm(r_ij, dim=1, keepdim=True)
		# Scale the distance by dividing the distance tensor by the Euclidean norm tensor (The biggest value will be 1 and lowest 0)
		dir_ij = r_ij / d_ij
		# phi_ij = self.radial_basis(d_ij)
		# What property of distance is better to keep for the model?
		# fcut = self.cutoff_fn(d_ij)
		# pprint(f"dir_ij: {dir_ij}")
		# pprint(f"d_ij is : {d_ij}")
		# pprint(f"fcut is : {fcut}")
		# filters = self.filter_net(phi_ij) * fcut[..., None]

		# if self.share_filters:
		#     filter_list = [filters] * self.n_interactions
		# else:
		#     filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

		# compute initial embeddings (they will learn during training from features)
		# Do I need to use this embedding for the model?
		q = self.embedding(atomic_numbers)[:, None]
		pprint(q)
		pprint(q.size())

		# add spin and charge embeddings
		# if hasattr(self, "activate_charge_spin_embedding") and self.activate_charge_spin_embedding:
		#     # get tensors from input dictionary
		#     pprint("**********START**********")
		#     total_charge = inputs[properties.total_charge]
		#     spin = inputs[properties.spin_multiplicity]
		#     print(spin)
		#     num_batch = len(inputs[properties.idx])
		#     print(num_batch)
		#     idx_m = inputs[properties.idx_m]
		#     print(idx_m)
		#     print("**********PASS**********")
		#
		#     charge_embedding = self.charge_embedding(
		#         q.squeeze(),
		#         total_charge,
		#         num_batch,
		#         idx_m
		#     )[:, None]
		#     spin_embedding = self.spin_embedding(
		#         q.squeeze(), spin, num_batch, idx_m
		#     )[:, None]
		#
		#     # additive combining of nuclear, charge and spin embedding
		#     q = (q + charge_embedding + spin_embedding)

		# compute interaction blocks and update atomic embeddings
		q_s = q.shape
		mu = torch.zeros((q_s[0], 3, q_s[2]), device=q.device)
		pprint(q.size())
		pprint(mu.size())
		pprint(dir_ij.size())
		# Concatenate the tensors dir_ij and q along the second axis
		conc_mu = torch.cat([dir_ij.unsqueeze(-1), mu], dim=-1)
		# pprint(conc_mu.size())
		self.feature_extractor_scalar(conc_mu)
		# Concatenate the tensors dir_ij and q along the second axis
		# pprint(q.size())
		# pprint(dir_ij.size())
		conc_q = torch.cat([dir_ij.unsqueeze(1), q], dim=-1)
		# pprint(conc_q.size())
		self.feature_extractor_vector(conc_q)
		# for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
		#     q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
		#     q, mu = mixing(q, mu)
		q = q.squeeze(1)
		# collect results
		inputs["scalar_representation"] = q
		inputs["vector_representation"] = mu

		return inputs


if __name__ == "__main__":
	import torch
	import schnetpack.nn as snn
	import schnetpack.properties as properties


	def test_painn_model():
		# Initialize a PaiNN model
		model = MLP(
			n_atom_basis=128,
			n_interactions=3,
			radial_basis=snn.GaussianRBF(n_rbf=20, cutoff=5.0),
			cutoff_fn=snn.CosineCutoff(5.0),
		)

		# Generate some random inputs
		inputs = {
			properties.Z: torch.randint(1, 101, (10,)),
			properties.Rij: torch.randn((10, 3)),
			properties.idx_i: torch.randint(0, 10, (10,)),
			properties.idx_j: torch.randint(0, 10, (10,)),
			properties.total_charge: torch.randn((10,)),
			properties.spin_multiplicity: torch.randn((10,)),
			properties.idx: torch.randint(0, 10, (10,)),
			properties.idx_m: torch.randint(0, 10, (10,)),
		}

		# Process the inputs through the model
		outputs = model(inputs)

		# Check if the outputs have the expected keys
		assert "scalar_representation" in outputs
		assert "vector_representation" in outputs

		# Check if the outputs have the expected shapes
		assert outputs["scalar_representation"].shape == (10, 128)
		assert outputs["vector_representation"].shape == (10, 3, 128)


	test_painn_model()
