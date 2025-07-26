import os
import subprocess
import tempfile
from typing import List, Tuple
import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
from rdkit import RDLogger
from rdkit.Chem import rdDistGeom
from useful_rdkit_utils import get_center
from tqdm import tqdm
# from synflownet.tasks.config import VinaConfig
import pickle
import time
import uuid
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logger = RDLogger.logger()
RDLogger.DisableLog("rdApp.*")



def read_pdbqt(fn):
    """
    Read a pdbqt file and return the RDKit molecule object.

    Args:
        - fn (str): Path to the pdbqt file.

    Returns:
        - mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.
    """
    pdbqt_mol = PDBQTMolecule.from_file(fn, is_dlg=False, skip_typing=True)
    rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    return rdkitmol_list[0]


def smile_to_conf(smile: str, n_tries=5) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)

    tries = 0
    while tries < n_tries:

        params = rdDistGeom.ETKDGv3()

        # set the parameters
        params.useSmallRingTorsions = True
        params.randomSeed = 0
        params.numThreads = 1

        # generate the conformer
        rdDistGeom.EmbedMolecule(mol, params)

        # add hydrogens
        mol = Chem.AddHs(mol, addCoords=True)

        if mol.GetNumConformers() > 0:
            return mol

        tries += 1

    print(f"Failed to generate conformer for {smile}")
    return mol


def mol_to_pdbqt(mol: Chem.Mol, pdbqt_file: str):

    # lg = RDLogger.logger()
    # lg.setLevel(RDLogger.ERROR)

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)

    for setup in mol_setups:
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
        if is_ok:
            with open(pdbqt_file, "w") as f:
                f.write(pdbqt_string)
            break
        else:
            print(f"Failed to write pdbqt file: {error_msg}")


def parse_affinty_from_pdbqt(pdbqt_file: str) -> float:
    with open(pdbqt_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "REMARK VINA RESULT" in line:
            return float(line.split()[3])
    return None


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
DATA_DIR = os.path.join(repo_root, "VinaAL/docking/")
# print(f'gpuvina.py script {script_dir} \n repo {repo_root} \n data dir {DATA_DIR}')

# TARGETS = {
#     "2bm2": {
#         "receptor": os.path.join(DATA_DIR, "2bm2/2bm2_protein.pdbqt"),
#         "center_x": 40.415,
#         "center_y": 110.986,
#         "center_z": 82.673,
#         "size_x": 30,
#         "size_y": 30,
#         "size_z": 30,
#         "num_atoms": 30,
#     },
#     "8azr": {
#         "receptor": os.path.join(DATA_DIR, "8azr/8azr.pdbqt"),
#         # "ref_ligand": os.path.join(DATA_DIR, "kras/8azr_ref_ligand.sdf"),
#         "center_x": 21.466,
#         "center_y": -0.650,
#         "center_z": 5.028,
#         "size_x": 18,
#         "size_y": 18,
#         "size_z": 18,
#         "num_atoms": 32,
#     },
#     "trmd": {
#         "receptor": os.path.join(DATA_DIR, "trmd/6qrd.pdbqt"),
#         "center_x": 16.957,
#         "center_y": 21.772,
#         "center_z": 33.296,
#         "size_x": 30,
#         "size_y": 30,
#         "size_z": 30,
#         "num_atoms": 34,
#     },
#      "fa7": {
#         "receptor": os.path.join(DATA_DIR, "fa7/fa7.pdbqt"),
#         "center_x": 10.131,
#         "center_y": 41.879,
#         "center_z": 32.097,
#         "size_x": 20.673,
#         "size_y": 20.198,
#         "size_z": 21.362,
#     },
#     "parp1": {
#         "receptor": os.path.join(DATA_DIR, "parp1/parp1.pdbqt"),
#         "center_x": 26.413,
#         "center_y": 11.282,
#         "center_z": 27.238,
#         "size_x": 18.521,
#         "size_y": 17.479,
#         "size_z": 19.995,
#     },
#     "5ht1b": {
#         "receptor": os.path.join(DATA_DIR, "5ht1b/5ht1b.pdbqt"),
#         "center_x": -26.602,
#         "center_y": 5.277,
#         "center_z": 17.898,
#         "size_x": 22.5,
#         "size_y": 22.5,
#         "size_z": 22.5,
#     },
#     "jak2": {
#         "receptor": os.path.join(DATA_DIR, "jak2/jak2.pdbqt"),
#         "center_x": 114.758,
#         "center_y": 65.496,
#         "center_z": 11.345,
#         "size_x": 19.033,
#         "size_y": 17.929,
#         "size_z": 20.283,
#     },
#     "braf": {
#         "receptor": os.path.join(DATA_DIR, "braf/braf.pdbqt"),
#         "center_x": 84.194,
#         "center_y": 6.949,
#         "center_z": -7.081,
#         "size_x": 22.032,
#         "size_y": 19.211,
#         "size_z": 14.106,
#     },
# }

TARGETS = {
    "2bm2": {
        "receptor": os.path.join(DATA_DIR, "2bm2/2bm2_protein.pdbqt"),
        "center_x": 40.415,
        "center_y": 110.986,
        "center_z": 82.673,
        "size_x": 30,
        "size_y": 30,
        "size_z": 30,
        "num_atoms": 30,
    },
    "8azr": {
        "receptor": os.path.join(DATA_DIR, "8azr/8azr.pdbqt"),
        "center_x": 21.466,
        "center_y": -0.650,
        "center_z": 5.028,
        "size_x": 18,
        "size_y": 18,
        "size_z": 18,
        "num_atoms": 32,
    },
    "trmd": {
        "receptor": os.path.join(DATA_DIR, "trmd/6qrd.pdbqt"),
        "center_x": 16.957,
        "center_y": 21.772,
        "center_z": 33.296,
        "size_x": 30,
        "size_y": 30,
        "size_z": 30,
        "num_atoms": 34,
    },
     "fa7": {
        "receptor": os.path.join(DATA_DIR, "fa7/fa7.pdbqt"),
        "center_x": 10.131,
        "center_y": 41.879,
        "center_z": 32.097,
        "size_x": 20.673,
        "size_y": 20.198,
        "size_z": 21.362,
    },
    "parp1": {
        "receptor": os.path.join(DATA_DIR, "parp1/parp1.pdbqt"),
        "center_x": 26.413,
        "center_y": 11.282,
        "center_z": 27.238,
        "size_x": 18.521,
        "size_y": 17.479,
        "size_z": 19.995,
    },
    "5ht1b": {
        "receptor": os.path.join(DATA_DIR, "5ht1b/5ht1b.pdbqt"),
        "center_x": -26.602,
        "center_y": 5.277,
        "center_z": 17.898,
        "size_x": 22.5,
        "size_y": 22.5,
        "size_z": 22.5,
    },
    "jak2": {
        "receptor": os.path.join(DATA_DIR, "jak2/jak2.pdbqt"),
        "center_x": 114.758,
        "center_y": 65.496,
        "center_z": 11.345,
        "size_x": 19.033,
        "size_y": 17.929,
        "size_z": 20.283,
    },
    "braf": {
        "receptor": os.path.join(DATA_DIR, "braf/braf.pdbqt"),
        "center_x": 84.194,
        "center_y": 6.949,
        "center_z": -7.081,
        "size_x": 22.032,
        "size_y": 19.211,
        "size_z": 14.106,
    },
    # New targets
    "1t7r": {
        "receptor": '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/1t7r.pdbqt',
        "center_x": 5.9246283,
        "center_y": 27.085604,
        "center_z": 44.94283,
        "size_x": 55.834,
        "size_y": 48.030003,
        "size_z": 60.945,
    },
    "5l2s": {
        "receptor": '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_ADT/5l2s.pdbqt', #'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/5l2s.pdbqt',
        "center_x":  13.844761,
        "center_y": 33.23798,
        "center_z": -0.20716855,
        "size_x": 30, #63.757004,
        "size_y": 30, #40.626,
        "size_z": 30,# 49.797,
    },
    "4f8h": {
        "receptor": '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/4f8h.pdbqt',
        "center_x": 47.319103,
        "center_y": -28.17705,
        "center_z": 26.800219,
        "size_x": 98.647,
        "size_y": 79.103004,
        "size_z": 122.267,
    },
    "6d6t": {
        "receptor": '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/6d6t.pdbqt',
        "center_x": 133.44765,
        "center_y": 147.10973,
        "center_z": 133.40134,
        "size_x": 133.491,
        "size_y": 121.785,
        "size_z": 126.274994,
    },
    "5mzj": {
        "receptor": '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/5mzj.pdbqt',
        "center_x": -17.796684,
        "center_y": -15.804938,
        "center_z": 18.386932,
        "size_x": 64.838,
        "size_y": 105.578,
        "size_z": 44.413998,
    },
    "4r06": {
        "receptor": '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/4r06.pdbqt',
        "center_x": 21.643887,
        "center_y": 11.027937,
        "center_z": 25.256802,
        "size_x": 68.372,
        "size_y": 66.567,
        "size_z": 72.105995,
    },
    "6iiu": {
        "receptor": '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/6iiu.pdbqt',
        "center_x": 19.85929,
        "center_y": 160.42876,
        "center_z": 142.18127,
        "size_x": 65.743996,
        "size_y": 98.315,
        "size_z": 61.10701,
    },
    "2zv2": {
        "receptor": '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/2zv2.pdbqt',
        "center_x": 1.6981819,
        "center_y": -14.587788,
        "center_z": -17.061022,
        "size_x": 45.721,
        "size_y": 45.497,
        "size_z": 61.336,
    },
    "4ag8": {
        "receptor": '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/4ag8.pdbqt',
        "center_x": 20.1937,
        "center_y": 23.861092,
        "center_z": 29.704456,
        "size_x": 54.75,
        "size_y": 51.833,
        "size_z": 57.178,
    },
    "4yay": {
        "receptor": '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/4yay.pdbqt',
        "center_x": -10.515906,
        "center_y": 10.254161,
        "center_z": 42.080994,
        "size_x": 59.577,
        "size_y": 50.473,
        "size_z": 89.836,
    },
    "1err": {
        "receptor": '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/1err.pdbqt',
        "center_x": 50.11242,
        "center_y": 37.065666,
        "center_z": 68.61871,
        "size_x": 67.253006,
        "size_y": 59.625,
        "size_z": 50.874,
    },
    "5ek0": {
        "receptor": '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/5ek0.pdbqt',
        "center_x": -47.85161,
        "center_y": -24.259432,
        "center_z": 2.4692063,
        "size_x": 112.405,
        "size_y": 75.6,
        "size_z": 108.65,
    },
}
# TARGETS = {
#     "2bm2": {
#         "receptor": os.path.join(DATA_DIR, "2bm2/2bm2_protein.pdbqt"),
#         "center_x": 40.415,
#         "center_y": 110.986,
#         "center_z": 82.673,
#         "size_x": 30,
#         "size_y": 30,
#         "size_z": 30,
#         "num_atoms": 30,
#     },
#     "8azr": {
#         "receptor": os.path.join(DATA_DIR, "8azr/8azr.pdbqt"),
#         "center_x": 21.466,
#         "center_y": -0.650,
#         "center_z": 5.028,
#         "size_x": 18,
#         "size_y": 18,
#         "size_z": 18,
#         "num_atoms": 32,
#     },
#     "trmd": {
#         "receptor": os.path.join(DATA_DIR, "trmd/6qrd.pdbqt"),
#         "center_x": 16.957,
#         "center_y": 21.772,
#         "center_z": 33.296,
#         "size_x": 30,
#         "size_y": 30,
#         "size_z": 30,
#         "num_atoms": 34,
#     },
#     "fa7": {
#         "receptor": os.path.join(DATA_DIR, "fa7/fa7.pdbqt"),
#         "center_x": 10.131,
#         "center_y": 41.879,
#         "center_z": 32.097,
#         "size_x": 20.673,
#         "size_y": 20.198,
#         "size_z": 21.362,
#     },
#     "parp1": {
#         "receptor": os.path.join(DATA_DIR, "parp1/parp1.pdbqt"),
#         "center_x": 26.413,
#         "center_y": 11.282,
#         "center_z": 27.238,
#         "size_x": 18.521,
#         "size_y": 17.479,
#         "size_z": 19.995,
#     },
#     "5ht1b": {
#         "receptor": os.path.join(DATA_DIR, "5ht1b/5ht1b.pdbqt"),
#         "center_x": -26.602,
#         "center_y": 5.277,
#         "center_z": 17.898,
#         "size_x": 22.5,
#         "size_y": 22.5,
#         "size_z": 22.5,
#     },
#     "jak2": {
#         "receptor": os.path.join(DATA_DIR, "jak2/jak2.pdbqt"),
#         "center_x": 114.758,
#         "center_y": 65.496,
#         "center_z": 11.345,
#         "size_x": 19.033,
#         "size_y": 17.929,
#         "size_z": 20.283,
#     },
#     "braf": {
#         "receptor": os.path.join(DATA_DIR, "braf/braf.pdbqt"),
#         "center_x": 84.194,
#         "center_y": 6.949,
#         "center_z": -7.081,
#         "size_x": 22.032,
#         "size_y": 19.211,
#         "size_z": 14.106,
#     },
#     "1t7r": {
#         "receptor": f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/1t7r.pdbqt',
#         "center_x": 10.134999,
#         "center_y": 24.233,
#         "center_z": 44.1815,
#         "size_x": 65.834,
#         "size_y": 58.030003,
#         "size_z": 70.945,
#     },
#     "5l2s": {
#         "receptor": f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/5l2s.pdbqt',
#         "center_x": 14.868501,
#         "center_y": 31.306,
#         "center_z": 1.9514999,
#         "size_x": 73.757,
#         "size_y": 50.626,
#         "size_z": 59.797,
#     },
#     "4f8h": {
#         "receptor": f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/4f8h.pdbqt',
#         "center_x": 44.497498,
#         "center_y": -27.265501,
#         "center_z": 25.6685,
#         "size_x": 108.647,
#         "size_y": 89.103004,
#         "size_z": 132.267,
#     },
#     "6d6t": {
#         "receptor": f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/6d6t.pdbqt',
#         "center_x": 121.315506,
#         "center_y": 141.9505,
#         "center_z": 125.9675,
#         "size_x": 143.491,
#         "size_y": 131.785,
#         "size_z": 136.275,
#     },
#     "5mzj": {
#         "receptor": f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/5mzj.pdbqt',
#         "center_x": -11.942,
#         "center_y": -23.485,
#         "center_z": 17.956,
#         "size_x": 74.838,
#         "size_y": 115.578,
#         "size_z": 54.413998,
#     },
#     "4r06": {
#         "receptor": f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/4r06.pdbqt',
#         "center_x": 24.094,
#         "center_y": 12.905499,
#         "center_z": 24.810999,
#         "size_x": 78.372,
#         "size_y": 76.567,
#         "size_z": 82.105995,
#     },
#     "6iiu": {
#         "receptor": f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/6iiu.pdbqt',
#         "center_x": 17.817999,
#         "center_y": 156.1945,
#         "center_z": 141.4065,
#         "size_x": 75.743996,
#         "size_y": 108.315,
#         "size_z": 71.10701,
#     },
#     "2zv2": {
#         "receptor": f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/2zv2.pdbqt',
#         "center_x": 5.9014997,
#         "center_y": -14.748501,
#         "center_z": -20.332,
#         "size_x": 55.721,
#         "size_y": 55.497,
#         "size_z": 71.336,
#     },
#     "4ag8": {
#         "receptor": f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/4ag8.pdbqt',
#         "center_x": 18.991001,
#         "center_y": 23.643501,
#         "center_z": 31.48,
#         "size_x": 64.75,
#         "size_y": 61.833,
#         "size_z": 67.178,
#     },
#     "4yay": {
#         "receptor": f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/4yay.pdbqt',
#         "center_x": -10.6625,
#         "center_y": 12.394501,
#         "center_z": 38.494003,
#         "size_x": 69.576996,
#         "size_y": 60.473,
#         "size_z": 99.836,
#     },
#     "1err": {
#         "receptor": '/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_ADT/1err.pdbqt',#f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/1err.pdbqt',
#         "center_x": 54.4945,
#         "center_y": 37.4815,
#         "center_z": 69.34,
#         "size_x": 40, # 77.253006,
#         "size_y": 40, #69.625,
#         "size_z": 40, #60.874,
#     },
#     "5ek0": {
#         "receptor": f'/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/docking/DD_targets_pdb/5ek0.pdbqt',
#         "center_x": -52.372498,
#         "center_y": -31.907,
#         "center_z": 3.9169998,
#         "size_x": 122.405,
#         "size_y": 85.6,
#         "size_z": 118.65,
#     },
# }


class QuickVina2GPU(object):

    def __init__(
        self,
        vina_path: str,
        target: str = None,
        target_pdbqt: str = None,
        reference_ligand: str = None,
        input_dir: str = None,
        out_dir: str = None,
        save_confs: bool = False,
        reward_scale_max: float = -1.0,
        reward_scale_min: float = -10.0,
        thread: int = 8000, #1000,
        print_time: bool = False,
        print_logs: bool = False,
    ):
        """
        Initializes the QuickVina2GPU class with configuration for running QuickVina 2 on GPU.

        Give either a code for a target or a PDBQT file.

        Args:
            - vina_path (str): Path to the Vina executable.
            - target (str, optional): Target identifier. Defaults to None.
            - target_pdbqt (str, optional): Path to the target PDBQT file. Defaults to None.
            - reference_ligand (str, optional): Path to the reference ligand file. Defaults to None.
            - input_dir (str, optional): Directory for input files. Defaults to a temporary directory.
            - out_dir (str, optional): Directory for output files. Defaults to None, will use input_dir + '_out'.
            - save_confs (bool, optional): Whether to save conformations. Defaults to False.
            - reward_scale_max (float, optional): Maximum reward scale. Defaults to -1.0.
            - reward_scale_min (float, optional): Minimum reward scale. Defaults to -10.0.
            - thread (int, optional): Number of threads to use. Defaults to 8000.
            - print_time (bool, optional): Whether to print execution time. Defaults to True.

        Raises:
        - ValueError: If the target is unknown.

        """
        self.vina_path = vina_path
        self.save_confs = save_confs
        self.thread = thread
        self.print_time = print_time
        self.print_logs = print_logs
        self.reward_scale_max = reward_scale_max
        self.reward_scale_min = reward_scale_min

        if target is None and target_pdbqt is None:
            raise ValueError("Either target or target_pdbqt must be provided")

        if input_dir is None:
            input_dir = tempfile.mkdtemp()
        else:
            temp_id   = uuid.uuid4().hex[:8]
            input_dir   = os.path.join(input_dir, temp_id)
            os.makedirs(input_dir, exist_ok=True) 

        self.input_dir = input_dir
        self.out_dir = input_dir + "_out"

        if target.lower() in TARGETS:
            self.target_info = TARGETS[target.lower()]
        else:
            raise ValueError(f"Unknown target: {target}")

        for key, value in self.target_info.items():
            setattr(self, key, value)

    def _write_config_file(self):

        config = []
        config.append(f"receptor = {self.receptor}")
        config.append(f"ligand_directory = {self.input_dir}")
        config.append(f"opencl_binary_path = /groups/cherkasvgrp/Vina-GPU-2.1/QuickVina2-GPU-2.1/") #{VinaConfig.opencl_binary_path}")
        config.append(f"center_x = {self.center_x}")
        config.append(f"center_y = {self.center_y}")
        config.append(f"center_z = {self.center_z}")
        config.append(f"size_x = {self.size_x}")
        config.append(f"size_y = {self.size_y}")
        config.append(f"size_z = {self.size_z}")
        config.append(f"thread = {self.thread}")

        with open(os.path.join(self.input_dir, "config.txt"), "w") as f:
            f.write("\n".join(config))
        # print(os.path.join(self.input_dir, "../config.txt"))
        # print(config)

    def _write_pdbqt_files(self, smiles: List[str]):

        # Convert smiles to mols
        mols = [smile_to_conf(smile, n_tries=3) for smile in tqdm(smiles, desc="Smiles conformation calculation")]

        
        # Remove None
        # mols = [mol for mol in mols if mol is not None]

        # Write pdbqt files
        for i, mol in enumerate(mols):
            pdbqt_file = os.path.join(self.input_dir, f"input_{i}.pdbqt")
            try:
                mol_to_pdbqt(mol, pdbqt_file)
            except Exception as e:
                print(f"Failed to write pdbqt file: {e}")

    def _teardown(self):
        # Remove input files
        if os.path.exists(self.input_dir): 
            for file in os.listdir(self.input_dir):
                os.remove(os.path.join(self.input_dir, file))
            os.rmdir(self.input_dir)

        # Remove output files
        if os.path.exists(self.out_dir):
            for file in os.listdir(self.out_dir):
                os.remove(os.path.join(self.out_dir, file))
            os.rmdir(self.out_dir)

    def _run_vina(self):

        result = subprocess.run(
            [self.vina_path, "--config", os.path.join(self.input_dir, "config.txt")], capture_output=True, text=True
        )
        if self.print_time:
            print(result.stdout.split("\n")[-2])
        if self.print_logs:
            print(result.stdout.split("\n"))

        if result.returncode != 0:
            print(f"Vina failed with return code {result.returncode}")
            print(result.stderr)
            return False

    def _parse_results(self):

        results = []
        failed = 0

        for i in range(self.batch_size):
            pdbqt_file = os.path.join(self.out_dir, f"input_{i}_out.pdbqt")
            if os.path.exists(pdbqt_file):
                affinity = parse_affinty_from_pdbqt(pdbqt_file)
            else:
                affinity = 0.0
                failed += 1
            results.append((affinity))

        if failed > 0:
            print(f"WARNING: Failed to calculate affinity for {failed}/{self.batch_size} molecules")

        return results

    def _parse_docked_poses(self):
        poses = []
        failed = 0

        for i in range(self.batch_size):
            pdbqt_file = os.path.join(self.out_dir, f"input_{i}_out.pdbqt")
            if os.path.exists(pdbqt_file):
                mol = read_pdbqt(pdbqt_file)
                poses.append(mol)
            else:
                poses.append(None)
                failed += 1

        if failed > 0:
            print(f"WARNING: Failed to read docked pdbqt files for {failed}/{self.batch_size} molecules")

        return poses

    def _check_outputs(self):
        if not os.path.exists(self.out_dir):
            return False
        return True

    def dock_mols(self, smiles: List[str]) -> List[Tuple[str, float]]:
        self.batch_size = len(smiles)

        # Write input files, config file and run vina
        self._write_pdbqt_files(smiles)
        self._write_config_file()
        self._run_vina()

        # Parse results
        affinties = self._parse_results()

        # Scale affinities to calculate rewards
        affinties = np.array(affinties)
        mols = self._parse_docked_poses()

        # print(
        #     f"AFFINITIES: mean={round(np.mean(affinties), 3 )}, std={round(np.std(affinties), 3)}, min={round(np.min(affinties), 3)}, max={round(np.max(affinties), 3)}"
        # )

        # Remove output files
        # self._teardown()

        return mols, affinties
    
    def dock_smiles_in_batches(self, smiles_list, batch_size=32):
        all_mols = []
        all_affinities = []

        # Splitting the SMILES list into batches
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            
            # Run vina.dock_mols on each batch
            mols, affinities = self.dock_mols(batch)
            
            # Append results to the aggregated list
            all_mols.extend(mols)
            all_affinities.extend(affinities)
        
        # Final aggregated results
        print(f"Docking completed for {len(smiles_list)} molecules.")
        print(f"Overall Affinities: mean={np.mean(all_affinities):.3f}, std={np.std(all_affinities):.3f}, min={np.min(all_affinities):.3f}, max={np.max(all_affinities):.3f}")
        
        return all_mols, all_affinities


def get_vina_scores(new_data, cursor, vina, num_gpus=1):
    mol_id_list = [item[0] for item in new_data]
    placeholders = ','.join('?' for _ in mol_id_list)
    cursor.execute(f'''
            SELECT smiles FROM molecules
            WHERE zinc_id IN ({placeholders})
        ''', mol_id_list)
    results = cursor.fetchall()
    smiles_list = [row[0] for row in results]
    docking_res = vina.dock_mols(smiles_list)
    # docking_res = vina.dock_smiles_in_batches(smiles_list)
    # print('gpuvina.py docking_res[0:5]',docking_res[1][0:5])
    assert len(mol_id_list)==len(smiles_list)==len(docking_res[1])
    return docking_res[1]


import subprocess
import os
import time
import shutil

# def get_vina_scores_mul_gpu(smiles_list, cursor, config, num_gpus=1, output_dir="vina_results", dockscore_gt=None):
#     """
#     Get docking scores for molecules by running separate Slurm jobs on each GPU.

#     :param new_data: List of tuples containing molecule IDs.
#     :param cursor: Database cursor to fetch SMILES.
#     :param vina: Vina docking object (for configuration purposes).
#     :param num_gpus: Number of GPUs to use for docking.
#     :param output_dir: Directory to save intermediate results from each GPU.
#     :param dockscore_gt: Precomputed docking score smiles_2_dockscore dictionary 
#     :return: List of docking scores.
#     """
    
#     # mol_id_list = [item[0] for item in new_data]

#     # smiles_list = []
#     # for i in tqdm(range(0, len(mol_id_list), 9999),desc='Retreiving SMILES from DB in gpuvina.py'):
#     #     batch = mol_id_list[i:i + 9999]
#     #     placeholders = ",".join("?" for _ in batch)  # Create placeholders for the query
#     #     query = f'''
#     #         SELECT smiles FROM molecules
#     #         WHERE zinc_id IN ({placeholders})
#     #     '''
#     #     cursor.execute(query, batch)
#     #     smiles_list.extend(row[0] for row in cursor.fetchall())

#     # placeholders = ','.join('?' for _ in mol_id_list)
#     # cursor.execute(f'''
#     #         SELECT smiles FROM molecules
#     #         WHERE zinc_id IN ({placeholders})
#     #     ''', mol_id_list)
#     # results = cursor.fetchall()
#     # smiles_list = [row[0] for row in results]
#     # mol_id_list = [item[0] for item in new_data]
#     # placeholders = ','.join('?' for _ in mol_id_list)
#     # cursor.execute(f'''
#     #         SELECT smiles FROM molecules
#     #         WHERE zinc_id IN ({placeholders})
#     #     ''', mol_id_list)
#     # results = cursor.fetchall()
#     # smiles_list = [row[0] for row in results]

#     # Create output directory
#     if os.path.exists(output_dir):
#         shutil.rmtree(output_dir)  # Remove the existing directory
#     os.makedirs(output_dir, exist_ok=True)  # Create a fresh directory

#     if dockscore_gt:
#         docking_res = [dockscore_gt[smiles] for smiles in smiles_list]
#         return docking_res
    
#     # Split SMILES list into chunks for each GPU
#     smiles_chunks = [smiles_list[i::num_gpus] for i in range(num_gpus)]

#     # Generate and submit Slurm jobs
#     job_ids = []
#     for gpu_id, chunk in enumerate(smiles_chunks):
#         chunk_file = os.path.join(output_dir, f"smiles_chunk_{gpu_id}.txt")
#         result_file = os.path.join(output_dir, f"results_{gpu_id}.pkl")
        
#         # Save the chunk to a file
#         with open(chunk_file, "w") as f:
#             for smiles in chunk:
#                 f.write(smiles + "\n")
        
#         # Create Slurm script for this chunk
#         slurm_script = os.path.join(output_dir, f"slurm_job_{gpu_id}.sh")
#         with open(slurm_script, "w") as f:
#             f.write(f"""#!/bin/bash
# #SBATCH --job-name=VnaGpu{gpu_id}
# #SBATCH --output={output_dir}/vina_gpu_{gpu_id}.out
# #SBATCH --partition=gpu-long
# #SBATCH --mem=100000M
# #SBATCH --gres=gpu:2
# #SBATCH --ntasks=1
# #SBATCH --time=102:00:00

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate dds 

# python /groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/VinaAL/run_vina.py --smiles_file {chunk_file} --output_file {result_file} --gpu_id {gpu_id} --target {config.global_params.target}
# """)
        
#         # Submit the Slurm job
#         job_id = subprocess.check_output(["sbatch", slurm_script]).decode("utf-8").strip().split()[-1]
#         job_ids.append(job_id)
#         time.sleep(0.1)

#     # Wait for all Slurm jobs to finish
#     # print(f"Submitted Slurm jobs: {job_ids}. Waiting for completion...")
#     prefix = "VnaGpu"
#     while True:
#         try:
#             # Get all running jobs for the current user
#             output = subprocess.check_output(["squeue", "-u", "$(whoami)"], shell=True).decode("utf-8").strip()
            
#             # Parse the output to filter jobs with the specified name prefix
#             jobs = [line for line in output.splitlines() if line.strip() and line.split()[2].startswith(prefix)]
            
#             if len(jobs) > 0:
#                 # print("Jobs still running:\n" )#+ "\n".join(jobs))
#                 time.sleep(30)  # Wait before checking again
#             else:
#                 print("All jobs with the specified name prefix have completed.")
#                 break
#         except subprocess.CalledProcessError:
#             # If squeue fails, assume there are no jobs with the prefix
#             print("All jobs with the specified name prefix have completed.")
#             break
#     # Combine results
#     docking_res = []
#     for gpu_id in range(num_gpus):
#         result_file = os.path.join(output_dir, f"results_{gpu_id}.pkl")
#         with open(result_file, "rb") as f:
#             docking_res.extend(pickle.load(f))
#     # print(docking_res)
#     assert len(smiles_list) == len(smiles_list) == len(docking_res)
#     return docking_res




# def get_glide_scores_mul_gpu(smiles_list, cursor, config, num_gpus=1, output_dir="glide_results", dockscore_gt=None):
#     run_conformer_job(iteration, 60, self.project_path, self.project_name, 'normal')
    
#     while True:
#         try:
#             # Get all running jobs for the current user
#             output = subprocess.check_output(["squeue", "-u", "$(whoami)"], shell=True).decode("utf-8").strip()
            
#             # Parse the output to filter jobs with the specified name prefix
#             jobs = [line for line in output.splitlines() if line.strip() and line.split()[2].startswith('phase_2')]
            
#             if len(jobs) > 0:
#                 time.sleep(30)  # Wait before checking again
#             else:
#                 print("All jobs with the specified name prefix have completed.")
#                 break
#         except subprocess.CalledProcessError:
#             print("All jobs with the specified name prefix have completed.")
#             break
        
#     # Step 3: Dock the molecules to get labels
#     labels = run_docking_job(iteration, 600, self.project_path, self.project_name, self.schrodinger_path, self.grid_file, self.glide_input_template)
    
#     while True:
#         try:
#             # Get all running jobs for the current user
#             output = subprocess.check_output(["squeue", "-u", "$(whoami)"], shell=True).decode("utf-8").strip()
            
#             # Parse the output to filter jobs with the specified name prefix
#             jobs = [line for line in output.splitlines() if line.strip() and line.split()[2].startswith('phase_3')]
            
#             if len(jobs) > 0:
#                 time.sleep(30)  # Wait before checking again
#             else:
#                 print("All jobs with the specified name prefix have completed.")
#                 break
#         except subprocess.CalledProcessError:
#             print("All jobs with the specified name prefix have completed.")
#             break
    
#     return labels
    
#     ######
#     while 'phase_2' in subprocess.getoutput('squeue -u mkpandey -h -o "%j"'): time.sleep(10)
#     #TODO: implement checking when step 2 is finished from squeue before starting docking.
    
#     # Step 3: Dock the molecules to get labels
#     labels = run_docking_job(iteration, 600, self.project_path, self.project_name, self.schrodinger_path, self.grid_file, self.glide_input_template)
#     return labels
#     while 'phase_3' in subprocess.getoutput('squeue -u mkpandey -h -o "%j"'): time.sleep(10)
#     pass

def get_vina_scores_mul_gpu(smiles_list, cursor, config, num_gpus=1, output_dir="vina_results", dockscore_gt=None):
    """
    Get docking scores for molecules by running separate Slurm jobs on each GPU.

    :param smiles_list: List of SMILES strings to dock.
    :param cursor: Database cursor (optional, for SMILES retrieval).
    :param config: Configuration object.
    :param num_gpus: Number of GPUs to use for docking.
    :param output_dir: Directory to save intermediate results from each GPU.
    :param dockscore_gt: Precomputed docking score SMILES-to-dockscore dictionary.
    :return: List of docking scores in the same order as the input SMILES list.
    """
    # Create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove the existing directory
    os.makedirs(output_dir, exist_ok=True)  # Create a fresh directory

    if dockscore_gt:
        docking_res = [dockscore_gt[smiles] for smiles in smiles_list]
        return docking_res

    # Attach indices to SMILES
    smiles_with_indices = [(i, smiles) for i, smiles in enumerate(smiles_list)]

    # Split SMILES list into chunks for each GPU
    smiles_chunks = [smiles_with_indices[i::num_gpus] for i in range(num_gpus)]

    # Generate and submit Slurm jobs
    job_ids = []
    for gpu_id, chunk in enumerate(smiles_chunks):
        chunk_file = os.path.join(output_dir, f"smiles_chunk_{gpu_id}.txt")
        result_file = os.path.join(output_dir, f"results_{gpu_id}.pkl")
        
        # Save the chunk (index and SMILES) to a file
        with open(chunk_file, "w") as f:
            for idx, smiles in chunk:
                f.write(f"{idx}\t{smiles}\n")
        
        # Create Slurm script for this chunk
        slurm_script = os.path.join(output_dir, f"slurm_job_{gpu_id}.sh") # get the run_vina.py. one job per gpu.
        with open(slurm_script, "w") as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name=VnaGpu{gpu_id}
#SBATCH --output={output_dir}/vina_gpu_{gpu_id}.out
#SBATCH --partition=gpu-long
#SBATCH --mem=100000M
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --time=102:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dds 

python run_vina.py --smiles_file {chunk_file} --output_file {result_file} --gpu_id {gpu_id} --target {config.global_params.target} --input_dir {config.global_params.dock_logs_path}
""")
        
        # Submit the Slurm job
        job_id = subprocess.check_output(["sbatch", slurm_script]).decode("utf-8").strip().split()[-1]
        job_ids.append(job_id)
        time.sleep(0.1)

    # Wait for all Slurm jobs to finish
    prefix = "VnaGpu"
    while True:
        try:
            # Get all running jobs for the current user
            output = subprocess.check_output(["squeue", "-u", "$(whoami)"], shell=True).decode("utf-8").strip()
            
            # Parse the output to filter jobs with the specified name prefix
            jobs = [line for line in output.splitlines() if line.strip() and line.split()[2].startswith(prefix)]
            
            if len(jobs) > 0:
                time.sleep(30)  # Wait before checking again
            else:
                print("All jobs with the specified name prefix have completed.")
                break
        except subprocess.CalledProcessError:
            print("All jobs with the specified name prefix have completed.")
            break

    # Combine and sort results
    docking_res = []
    for gpu_id in range(num_gpus):
        result_file = os.path.join(output_dir, f"results_{gpu_id}.pkl")
        with open(result_file, "rb") as f:
            docking_res.extend(pickle.load(f))  # Each result should be (index, score)
    
    # Sort results by index to match the original SMILES order
    # print('dock res ', docking_res)
    docking_res.sort(key=lambda x: x[0])
    sorted_scores = [score for _, score in docking_res]
    
    assert len(smiles_list) == len(sorted_scores)
    return sorted_scores
