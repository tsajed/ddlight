
import subprocess
import time

import os
import argparse
import pickle
import glob
# import sys
# sys.path.append('/groups/cherkasvgrp/Student_backup/mkpandey/My_Projects/DDSgroups/')

import glob
import csv

def write_smiles_to_file(smiles_list, lib_idx_list, file_path):
    with open(file_path, 'w') as f:
        for smile, lib_idx in zip(smiles_list, lib_idx_list):
            f.write(smile +' '+lib_idx+ '\n')

def write_job_id(protein, file_path, iteration, job_id, job_name):
    """
    Write the job ID and create necessary directories for the specified iteration.
    Mimics the behavior of the original jobid_writer.py script.
    """
    if iteration != -1:  # Create directory for the current iteration
        iteration_dir = os.path.join(file_path, protein, f'iteration_{iteration}')
        os.makedirs(iteration_dir, exist_ok=True)
        with open(os.path.join(iteration_dir, job_name), 'w') as f:
            f.write(f"{job_id}\n")
    else:  # For jobs that occur after an iteration
        after_iteration_dir = os.path.join(file_path, protein, 'after_iteration')
        os.makedirs(after_iteration_dir, exist_ok=True)
        with open(os.path.join(after_iteration_dir, job_name), 'w') as f:
            f.write(f"{job_id}\n")

def write_conf_script(file, name, num_cpus, sdf_dir):
    """Creates the bash script for running OE Omega."""
    script_content = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1

oeomega classic -in {file} -out {sdf_dir}/{name}_sdf.sdf -maxconfs 1 -strictstereo false -mpi_np {num_cpus} -log {name}.log -prefix {name} -warts false
"""
    script_file = os.path.join(os.path.dirname(sdf_dir),f'{name}_conf.sh')  # sdf_dir.rsplit('/', 2)[0] + '/'    #os.path.join(sdf_dir, f'{name}_conf.sh') #
    print('script_file file ', script_file)
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    return script_file

def run_conformer_job(batches, iteration, num_cpus, project_path, project_name, cpu_partition):

    ''' batches is EasyDict with keys 'train', 'validation', 'test'. Each key has smiles and libID attributes.
    batches.train.smiles is a list of SMILES strings.
    batches.train.libID is a list of corresponding library IDs.
    '''
    
    write_job_id(project_name, project_path, iteration, 'phase_2', 'phase_2.txt')

    # Set up directories and process files
    iteration_dir = os.path.join(project_path, project_name, f'iteration_{iteration}')
    sdf_dir = os.path.join(iteration_dir, 'sdf')
    os.makedirs(sdf_dir, exist_ok=True)

    # # Read the batch.pkl (aug_batch.pkl) file
    # batch_file = os.path.join(iteration_dir, 'aug_batch.pkl')
    # with open(batch_file, 'rb') as f:
    #     batches = pickle.load(f)  # Dictionary with keys 'train', 'validation', 'test'
    # # Iterate over the 'train', 'validation', and 'test' keys if iteration 0
    # # For other itearions 'val' and 'test' are same as before, hence only do for 'train'
    if iteration==0:
        for name, batch in batches.items():
            write_smiles_to_file(batch.smiles, batch.libID ,iteration_dir+f'/smile_{name}.smi')
    else:
        name = 'train'
        batch = batches['train']
        write_smiles_to_file(batch.smiles, batch.libID ,iteration_dir+f'/smile_{name}.smi')

  
    # Process SMILES files
    files = glob.glob(iteration_dir+'/smile_*.smi')
    print('conformers.py files ', files, 'iteration_dir ', iteration_dir)
    for file in files:
        name_prefix = file.split('/')[-1].split('_')[-1].split('.smi')[0]
        if iteration ==0:
            # Need train val test conformers only for training only for iteration 0
            # for iteration 1,2,..., val and test set remain the same as before
            if name_prefix == 'train':
                name = 'training'
            elif name_prefix == 'val' or name_prefix == 'validation':
                name = 'validation'
            elif name_prefix == 'test':
                name = 'testing'
            # else:
            #     continue
        else:
            if name_prefix == 'train':
                name = 'training'
            else:
                continue

        # Create the OE Omega bash script for this file
        script_file = write_conf_script(file, name, num_cpus, sdf_dir)

        # Submit the job using sbatch for the created script
        os.system(f"sbatch -J phase_2 -p {cpu_partition} -c {num_cpus} {script_file}")



def wait_for_jobs_to_finish(job_prefix, check_interval=30):
    """Wait until all SLURM jobs with a given prefix are finished."""
    while True:
        try:
            output = subprocess.check_output(
                ["squeue", "-u", "$(whoami)"], shell=True
            ).decode("utf-8").strip()
            jobs = [line for line in output.splitlines() 
                    if line.strip() and line.split()[2].startswith(job_prefix)]
            if jobs:
                time.sleep(check_interval)
            else:
                print(f"All jobs with prefix '{job_prefix}' have completed.")
                break
        except subprocess.CalledProcessError:
            print(f"All jobs with prefix '{job_prefix}' have completed (by exception).")
            break


##### Glide docking #####

class GlideInputPreparer:
    def __init__(self, project_name, file_path, grid_file, iteration_no, glide_input):
        self.project_name = project_name
        self.file_path = file_path
        self.grid_file = grid_file
        self.iteration_no = int(iteration_no)
        self.glide_input = glide_input

    def prepare_glide_input(self):
        """
        Prepares the Glide input files for docking by creating the necessary directories,
        copying templates, and replacing the placeholders with appropriate values.
        """
        # Create directories for the docked files
        if self.iteration_no != -1:
            docked_dir = os.path.join(self.file_path, self.project_name, f'iteration_{self.iteration_no}', 'docked')
        else:
            docked_dir = os.path.join(self.file_path, self.project_name, 'after_iteration', 'to_dock', 'docked')

        os.makedirs(docked_dir, exist_ok=True)

        # Set the path for ligand files
        if self.iteration_no != -1:
            ligand_file_pattern = os.path.join(self.file_path, self.project_name, f'iteration_{self.iteration_no}', 'sdf', '*')
        else:
            ligand_file_pattern = os.path.join(self.file_path, self.project_name, 'after_iteration', 'to_dock', 'sdf', '*')

        # Process each ligand file
        for ligand_file in glob.glob(ligand_file_pattern):
            if self.iteration_no != -1:
                name = ligand_file.split('/')[-1].split('_')[0] + '_docked'
            else:
                name = ligand_file.split('/')[-1].split('.')[0] + '_docked'

            if self.iteration_no != -1:
                output_file = os.path.join(self.file_path, self.project_name, f'iteration_{self.iteration_no}', 'docked', f'{name}.in')
            else:
                output_file = os.path.join(self.file_path, self.project_name, 'after_iteration', 'to_dock', 'docked', f'{name}.in')

            # Create the .in file by replacing placeholders with actual values
            ref1 =  open(output_file,'w')
            with open(self.glide_input,'r') as ref:
                for line in ref:
                    if 'GRIDFILE' in line:
                        ref1.write('GRIDFILE '+self.grid_file+'\n')
                    elif 'LIGANDFILE' in line:
                        ref1.write('LIGANDFILE '+ligand_file+'\n')
                    else:
                        ref1.write(line)
            ref1.close()

def prepare_glide(project_name, file_path, grid_file, iteration_no, glide_input_template):
    """
    Method to prepare Glide input files for docking. Can be imported and used in another file.

    Args:
        project_name (str): Name of the project.
        file_path (str): Path to the project folder.
        grid_file (str): Path to the docking grid file (zip).
        iteration_no (int): The current iteration number (-1 if after iterations).
        glide_input_template (str): Path to the template for Glide input file.
    """
    preparer = GlideInputPreparer(project_name, file_path, grid_file, iteration_no, glide_input_template)
    preparer.prepare_glide_input()
    
def run_glide(file_path, protein, iteration_no, schrodinger_path, njobs):
    """
    Runs Glide docking jobs using the SCHRODINGER software for the specified iteration.
    
    Args:
        file_path (str): The base file path where the project is located.
        protein (str): The protein name for the project.
        iteration_no (int): The current iteration number.
        schrodinger_path (str): The path to the SCHRODINGER executable.
        njobs (int): Number of jobs to run in parallel.
    """
    # Define the working directory
    docked_dir = os.path.join(file_path, protein, f"iteration_{iteration_no}", "docked")
    
    # Change to the docked directory
    os.chdir(docked_dir)
    
    
    # Run the glide docking jobs for each .in file
    for in_file in glob.glob("*.in"):
        job_name = f"phase_3_{os.path.splitext(in_file)[0]}"
        
        # Construct the glide command
        glide_command = [
            os.path.join(schrodinger_path, "glide"),
            "-HOST", "slurm-compute",
            "-NJOBS", str(njobs),
            "-OVERWRITE",
            "-JOBNAME", job_name,
            in_file
        ]
        print('docking.py glide command ',glide_command)
        
        # Run the glide command using subprocess
        subprocess.run(glide_command, check=True)

def label_extracter(input_file, iteration_no, output_file):
    with open(input_file, 'r') as csv_file, open(output_file, 'w') as txt_file:
        reader = csv.DictReader(csv_file)
        
        # Write header to the output file
        txt_file.write('r_i_docking_score,title\n')
        
        # Process each row from the CSV
        for row in reader:
            docking_score = row['r_i_docking_score']
            title = row['title']
            
            # Write docking_score and title in the required format to labels.txt
            txt_file.write(f'{docking_score},{title}\n')

def extract_labels(file_path,project_name,iteration_no,score_keyword):
    if iteration_no ==0:
        # Need train val test conformers only for training only for iteration 0
        # for iteration 1,2,..., val and test set remain the same as before
        sets = ["training", "testing", "validation"]
    else:
        sets = ["training"]
    if score_keyword =='glide':
        for split in sets:
            input_file = f'{file_path}/{project_name}/iteration_{iteration_no}/docked/phase_3_{split}_docked.csv'  # Path to your input file
            output_file = f'{file_path}/{project_name}/iteration_{iteration_no}/{split}_labels.txt'  # Desired output file name
            if iteration_no>0:
                input_file_prev_it = f'{file_path}/{project_name}/iteration_{iteration_no-1}/docked/phase_3_{split}_docked.csv'  # Append docking results of previous iteration batch
            else:
                input_file_prev_it = None
            label_extracter(input_file, input_file_prev_it, output_file)
            print(f"{split}Labels extracted and written to {output_file}.")

def run_docking_job(iteration, num_cpus, project_path, project_name, schrodinger_path, grid_file, glide_input_template):
    write_job_id(project_name, project_path, iteration, 'phase_3', 'phase_3.txt')
    # Calculate the number of parallel jobs to run
    njobs = num_cpus // 3 if iteration ==0 else num_cpus
    prepare_glide(project_name,project_path,grid_file,iteration,glide_input_template)
    run_glide(project_path, project_name, iteration, schrodinger_path, njobs)
    while 'phase_3' in subprocess.getoutput('squeue -u mkpandey -h -o "%j"'): time.sleep(10)
    extract_labels(project_path,project_name,iteration,"glide")


def get_glide_scores_mul_gpu(batches, iteration, config, output_dir="glide_results", dockscore_gt=None):
    # Step 1: Run conformer generation
    run_conformer_job(batches, iteration, 60, config.global_params.project_path, config.global_params.project_name, 'normal')

    # Step 2: Wait for conformer generation jobs to complete
    wait_for_jobs_to_finish(job_prefix='phase_2')

    # Step 3: Run docking jobs
    run_docking_job(
        iteration, 600, config.global_params.project_path, config.global_params.project_name,
        config.global_params.schrodinger_path, config.global_params.grid_file, config.global_params.glide_input_template
    )

    # Step 4: Wait for docking jobs to complete
    wait_for_jobs_to_finish(job_prefix='phase_3')
    
    # get labels for batches
    train_id_to_dockscore = {}
    val_id_to_dockscore = {}
    test_id_to_dockscore = {}
    for split in ['training', 'validation', 'testing']:
        labels_file = os.path.join(config.global_params.project_path, config.global_params.project_name, f'iteration_{iteration}', f'{split}_labels.txt')
        with open(labels_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip the header line
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    score, lib_id = parts
                    if split == 'training':
                        train_id_to_dockscore[lib_id] = float(score)
                    elif split == 'validation':
                        val_id_to_dockscore[lib_id] = float(score)
                    elif split == 'testing':
                        test_id_to_dockscore[lib_id] = float(score)
                        
    train_docksores, val_docksores, test_docksores = [], [], []
    for split in batches.keys():
        batch = batches[split]
        if split == 'train':
            # print('glide_dock.py train_id_to_dockscore ', train_id_to_dockscore)
            # print('glide_dock.py batch.libID ', batch.libID)
            for libID in batch.libID:
                if libID in train_id_to_dockscore:
                    train_docksores.append(train_id_to_dockscore[libID])
                else:
                    train_docksores.append(0.0)
        if split == 'validation':
            for libID in batch.libID:
                if libID in val_id_to_dockscore:
                    val_docksores.append(val_id_to_dockscore[libID])
                else:
                    val_docksores.append(0.0)
        if split == 'test':
            for libID in batch.libID:
                if libID in test_id_to_dockscore:
                    test_docksores.append(test_id_to_dockscore[libID])
                else:
                    test_docksores.append(0.0)

    return train_docksores, val_docksores, test_docksores   