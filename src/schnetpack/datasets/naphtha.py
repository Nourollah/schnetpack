import io
import logging
import os
import re
import shutil
import tarfile
import tempfile
import pathlib
from typing import List, Optional, Dict
from urllib import request as request

import numpy as np
from ase import Atoms
from ase.io.extxyz import read_xyz
from tqdm import tqdm

import torch
from schnetpack.data import *
import schnetpack.properties as structure
from schnetpack.data import AtomsDataModuleError, AtomsDataModule

__all__ = ["Naphtha"]


class Naphtha(AtomsDataModule):
	"""QM9 benchmark database for organic molecules.

	The QM9 database contains small organic molecules with up to nine non-hydrogen atoms
	from including C, O, N, F. This class adds convenient functions to download QM9 from
	figshare and load the data into pytorch.

	"""

	# properties
	energy = "energy"
	time = "time"

	atomrefs = {
		energy: [
			0.0,
			-313.5150902000774,
			0.0,
			0.0,
			0.0,
			0.0,
			-23622.587180094913,
			-34219.46811826416,
			-47069.30768969713,
		]
	}

	def __init__(
			self,
			datapath: str,
			batch_size: int,
			xyz_path: pathlib.Path,
			num_train: Optional[int] = None,
			num_val: Optional[int] = None,
			num_test: Optional[int] = None,
			split_file: Optional[str] = "split.npz",
			format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
			load_properties: Optional[List[str]] = None,
			val_batch_size: Optional[int] = None,
			test_batch_size: Optional[int] = None,
			transforms: Optional[List[torch.nn.Module]] = None,
			train_transforms: Optional[List[torch.nn.Module]] = None,
			val_transforms: Optional[List[torch.nn.Module]] = None,
			test_transforms: Optional[List[torch.nn.Module]] = None,
			num_workers: int = 2,
			num_val_workers: Optional[int] = None,
			num_test_workers: Optional[int] = None,
			property_units: Optional[Dict[str, str]] = None,
			distance_unit: Optional[str] = None,
			data_workdir: Optional[str] = None,
			**kwargs
	):
		"""

		Args:
			datapath: path to dataset
			batch_size: (train) batch size
			num_train: number of training examples
			num_val: number of validation examples
			num_test: number of test examples
			split_file: path to npz file with data partitions
			format: dataset format
			load_properties: subset of properties to load
			remove_uncharacterized: do not include uncharacterized molecules.
			val_batch_size: validation batch size. If None, use test_batch_size, then batch_size.
			test_batch_size: test batch size. If None, use val_batch_size, then batch_size.
			transforms: Transform applied to each system separately before batching.
			train_transforms: Overrides transform_fn for training.
			val_transforms: Overrides transform_fn for validation.
			test_transforms: Overrides transform_fn for testing.
			num_workers: Number of data loader workers.
			num_val_workers: Number of validation data loader workers (overrides num_workers).
			num_test_workers: Number of test data loader workers (overrides num_workers).
			property_units: Dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
			distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
			data_workdir: Copy data here as part of setup, e.g. cluster scratch for faster performance.
		"""
		super().__init__(
			datapath=datapath,
			batch_size=batch_size,
			num_train=num_train,
			num_val=num_val,
			num_test=num_test,
			split_file=split_file,
			format=format,
			load_properties=load_properties,
			val_batch_size=val_batch_size,
			test_batch_size=test_batch_size,
			transforms=transforms,
			train_transforms=train_transforms,
			val_transforms=val_transforms,
			test_transforms=test_transforms,
			num_workers=num_workers,
			num_val_workers=num_val_workers,
			num_test_workers=num_test_workers,
			property_units=property_units,
			distance_unit=distance_unit,
			data_workdir=data_workdir,
			**kwargs
		)

		self.xyz_path = xyz_path

	def prepare_data(self):
		if not os.path.exists(self.datapath):
			property_unit_dict = {
				Naphtha.energy: "meV",
				# Naphtha.time: "mS",
			}

			tmpdir = tempfile.mkdtemp("naphtha")
			# atomrefs = self._download_atomrefs(tmpdir)

			dataset = create_dataset(
				datapath=self.datapath,
				format=self.format,
				distance_unit="Ang",
				property_unit_dict=property_unit_dict,
				atomrefs=Naphtha.atomrefs,
			)


			self._download_data(tmpdir, dataset, self.xyz_path)
			shutil.rmtree(tmpdir)
		else:
			dataset = load_dataset(self.datapath, self.format)


	def _download_data(
			self, tmpdir: pathlib.Path, dataset: BaseAtomsData, raw_path: pathlib.Path
	):

		logging.info("Parse xyz files...")
		ordered_files = sorted(
			os.listdir(raw_path), key=lambda x: (int(re.sub("\D", "", x)), x)
		)

		property_list = []

		irange = np.arange(len(ordered_files), dtype=int)

		# pattern = r"i\s*=\s*(\d+),\s*time\s*=\s*([\d.]+),\s*E\s*=\s*([-.\d]+)"
		pattern = r"E\s*=\s*(-\d+(?:\.\d+)?)"

		for i in tqdm(irange):
			xyzfile = os.path.join(raw_path, ordered_files[i])
			properties = {}

			tmp = io.StringIO()
			with open(xyzfile, "r") as f:
				lines = f.readlines()

				match = re.search(pattern, lines[1])
				# time = float(match.group(3))
				energy = float(match.group(1))

				for pn, p in zip(dataset.available_properties, [energy]):
					properties[pn] = np.array([p])
				for line in lines:
					tmp.write(line.replace("*^", "e"))

			tmp.seek(0)
			ats: Atoms = list(read_xyz(tmp, 0))[0]
			properties[structure.Z] = ats.numbers
			properties[structure.R] = ats.positions
			properties[structure.cell] = ats.cell
			properties[structure.pbc] = ats.pbc
			property_list.append(properties)

		logging.info("Write atoms to db...")
		dataset.add_systems(property_list=property_list)
		logging.info("Done.")
