import copy
import dataclasses
import io
import logging
import os
import re
import shutil
import tarfile
import tempfile
import typing
from typing import List, Optional, Dict
from urllib import request as request

import numpy as np
from ase import Atoms
from ase.io.extxyz import read_xyz
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

import torch
from schnetpack.data import *
import schnetpack.properties as structure
from schnetpack.data import AtomsDataModuleError, AtomsDataModule

__all__ = ["HOPV15"]


class HOPV15(AtomsDataModule):
    """HOPV15 database for quantum-chemical calculation.

    Refer to https://www.nature.com/articles/sdata201686#Tab1 for more information
    on the dataset.

    References:

        .. [#hopv_15] https://figshare.com/articles/dataset/HOPV15_Dataset/1610063

    """

    # properties
    A = "rotational_constant_A"
    B = "rotational_constant_B"
    C = "rotational_constant_C"

    @dataclasses.dataclass
    class ExperimentalInformation:
        doi: typing.Final[str] = "digital_object_identifier"
        inchikey: typing.Final[str] = "InChIKEY"
        construction: typing.Final[str] = "construction"
        architecture: typing.Final[str] = "architecture"
        complement: typing.Final[str] = "complement"
        homo: typing.Final[str] = "homo"
        lumo: typing.Final[str] = "lumo"
        e_gap: typing.Final[str] = "e_gap"
        o_gap: typing.Final[str] = "o_gap"
        pce: typing.Final[str] = "PCE"
        V_oc: typing.Final[str] = "V_oc"
        J_sc: typing.Final[str] = "J_sc"
        fill_factor: typing.Final[str] = "fill_factor"

    def __init__(
            self,
            datapath: str,
            batch_size: int,
            num_train: Optional[int] = None,
            num_val: Optional[int] = None,
            num_test: Optional[int] = None,
            split_file: Optional[str] = "split.npz",
            format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
            load_properties: Optional[List[str]] = None,
            remove_uncharacterized: bool = False,
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
            conformers_as_data: Optional[bool] = False,
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

        # conformers_as_data: If True, each conformer is treated as a separate molecule.
        self.conformers_as_data = conformers_as_data

        self.remove_uncharacterized = remove_uncharacterized

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            property_unit_dict = {
                HOPV15.ExperimentalInformation.homo: "Volts",
                HOPV15.ExperimentalInformation.lumo: "Volts",
                HOPV15.ExperimentalInformation.pce: "percent",
                HOPV15.ExperimentalInformation.e_gap: "Volts",
                HOPV15.ExperimentalInformation.o_gap: "mA/cm^2",
                # HOPV15.ExperimentalInformation.inchikey: "classes",
                # HOPV15.ExperimentalInformation.doi: "classes",
                # HOPV15.ExperimentalInformation.construction: "classes",
                # HOPV15.ExperimentalInformation.architecture: "classes",
                # HOPV15.ExperimentalInformation.complement: "classes",
                # HOPV15.ExperimentalInformation.V_oc: "Volts",
                # HOPV15.ExperimentalInformation.J_sc: "Volts",
                # HOPV15.ExperimentalInformation.fill_factor: "Volts",
                # "dft_outputs": "calculation",
            }

            tmpdir = tempfile.mkdtemp("hopv15")

            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
            )

            self._download_data(tmpdir, dataset)
            shutil.rmtree(tmpdir)
        else:
            dataset = load_dataset(self.datapath, self.format)

    def _download_data(
            self,
            tmpdir,
            dataset: BaseAtomsData,
            conformers_as_data: Optional[bool] = True
    ):
        logging.info("Downloading HOPV15 data...")
        raw_path = os.path.join(tmpdir, "HOPV_15_revised_2.data")
        url = "https://figshare.com/ndownloader/files/4513735"

        request.urlretrieve(url, raw_path)
        logging.info("Done.")

        logging.info("Parse data file...")
        self._parse_hopv15(
            dataset,
            raw_path,
            conformers_as_data=conformers_as_data)
        logging.info("Done.")

    def _parse_hopv15(
            self,
            dataset: BaseAtomsData,
            file_path: str,
            conformers_as_data: bool = True,
            single_conformer: bool = False
    ):
        with open(file_path, 'r') as f:
            lines: typing.List[typing.AnyStr] = f.readlines()

        # Initialize list to store all molecules and their properties
        if conformers_as_data:
            """Description of the process
            The dataset file is formed arbitrary so we cannot put it in a 
            ordinary loop. Based on our strategy we will use a while loop
            with a floating integer to move the indicator i inside the loop.
            This indicator will be used to move through the lines of the file.
            """
            property_list: typing.List[dict] = []
            atoms_list: typing.List[Atoms] = []
        else:
            property_list: typing.List[typing.Tuple[Atoms, dict]] = []
            atoms_list: typing.List[Atoms] = []

        i: int = 0

        # Iterate over lines in file
        while i < len(lines):
            # Parse SMILES string and convert to ase.Atoms object
            smiles: str = lines[i].strip()
            mol: str = Chem.MolFromSmiles(smiles)
            # mol: str = Chem.AddHs(mol)
            # AllChem.EmbedMolecule(mol)

            # Parse InChI string
            inchi: str = lines[i + 1].strip()

            # Parse references
            forces: typing.List[str] = lines[i + 2].strip().split(",")
            ei_properties: dict = {
                # "inChi": inchi, @TODO: Convert this unites to proper format (int, float, normalization, ...)
                # HOPV15.ExperimentalInformation.doi: forces[0],
                # HOPV15.ExperimentalInformation.inchikey: forces[1],
                # HOPV15.ExperimentalInformation.construction: forces[2],
                # HOPV15.ExperimentalInformation.architecture: forces[3],
                # HOPV15.ExperimentalInformation.complement: forces[4],
                HOPV15.ExperimentalInformation.homo: np.array([float(forces[5])], dtype=np.float64),
                HOPV15.ExperimentalInformation.lumo: np.array([float(forces[6])], dtype=np.float64),
                HOPV15.ExperimentalInformation.e_gap: np.array([float(forces[7])], dtype=np.float64),
                HOPV15.ExperimentalInformation.o_gap: np.array([float(forces[8])], dtype=np.float64),
                HOPV15.ExperimentalInformation.pce: np.array([float(forces[9])], dtype=np.float64),
                # HOPV15.ExperimentalInformation.V_oc: np.ndarray([float(forces[10])], dtype=np.float64),
                # HOPV15.ExperimentalInformation.J_sc: np.ndarray([float(forces[11])], dtype=np.float64),
                # HOPV15.ExperimentalInformation.fill_factor: np.ndarray([float(forces[12])], dtype=np.float64)
                }
            # Parse number of conformers
            num_conformers: int = int(lines[i + 4].strip())
            i += 5

            """Moving rule:
            Leave indicator at the conformer id (e.g leave it at line with
            information like Conformer 3)
            """

            # Parse atomic coordinates
            for conformer_number in range(num_conformers):
                # Extracting features of a conformer from the file
                conformer_id: int = int(lines[i].strip().split()[-1])
                atoms_lines: int = int(lines[i + 1].strip())
                # Breake for the non-considering different conformers
                if conformer_number > 1 and single_conformer is True:
                    i += atoms_lines + 6
                    continue
                conformer_atoms: typing.List[str] = []
                conformer_positions: typing.List[typing.Tuple[float, float, float]] = []
                for j in range(atoms_lines):
                    atom_name, x, y, z = [func(dtype) for func, dtype in
                                          zip((str, float, float, float), lines[i + 2 + j].split())]
                    conformer_atoms.append(atom_name)
                    conformer_positions.append((x, y, z))
                ats: Atoms = Atoms(conformer_atoms, positions=conformer_positions)
                if conformers_as_data:

                    molecule_properties: dict = copy.deepcopy(ei_properties)

                    i += atoms_lines + 2
                    # DFT Calculated Features
                    # @TODO: parse dft_output line in this part
                    # molecule_properties["dft_outputs"] = lines[i: i + 4]
                    i += 4

                    atoms_list.append(ats)
                    property_list.append(molecule_properties)

                else:
                    i += atoms_lines + 2
                    atoms_list.append(ats)
                    property_list.append(ei_properties)
            # @TODO: Addeing all the conformers as a one molecule here

        logging.info("Write atoms to db...")
        dataset.add_systems(property_list=property_list, atoms_list=atoms_list)
