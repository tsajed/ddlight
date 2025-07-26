import argparse
from gpuvina import QuickVina2GPU
import pickle

def main():
    parser = argparse.ArgumentParser(description="Run Vina docking on a chunk of SMILES.")
    parser.add_argument("--smiles_file", type=str, required=True, help="Path to the SMILES input file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file.")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID to use.")
    parser.add_argument("--target", type=str, required=True, help="PDB ID")
    parser.add_argument("--input_dir", type=str, required=False, default=None, help="The output Directory")
    args = parser.parse_args()

    # Initialize Vina
    vina = QuickVina2GPU(vina_path="/groups/cherkasvgrp/Vina-GPU-2.1/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1", #QuickVina2-GPU-2-1"', # Avoiding global initialization because _teardown deletes tmp dirs
                        target=args.target,
                        input_dir=args.input_dir)

    # Read SMILES
    with open(args.smiles_file, "r") as f:
        # smiles_list = [line.strip() for line in f]
        splitted_lines = [line.strip().split('\t') for line in f]
        smiles_list = [s[1] for s in splitted_lines]
        idx_list = [int(s[0]) for s in splitted_lines]

    # Run docking
    docking_res = vina.dock_mols(smiles_list)
    idx_dockres_tup = [(idx, dres) for idx, dres in zip( idx_list, docking_res[1])]
    
    print('runvina.py args.output_file ',args.output_file)
    print('runvina.py idx_dockres_tup', idx_dockres_tup)

    # Save results
    with open(args.output_file, "wb") as f:
        pickle.dump(idx_dockres_tup,f )#docking_res[1],f)

        # for score in docking_res:
        #     print('score ', score)
        #     f.write(f"{score}\n")

    vina._teardown()


if __name__ == "__main__":
    main()
