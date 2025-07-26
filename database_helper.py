# import numpy as np

# def molids_2_fps(cursor,mol_ids):
#     fingerprints = []

#     # Use placeholders to query multiple Zinc IDs
#     placeholders = ','.join('?' for _ in mol_ids)
    
#     # Fetch the sparse components for the Zinc IDs
#     cursor.execute(f'''
#         SELECT indices FROM molecules
#         WHERE zinc_id IN ({placeholders})
#     ''', mol_ids)
    
#     results = cursor.fetchall()
#     # print(results)
    
#     for indices_blob in results:
#         # Convert binary blobs back to their original arrays
#         indices = np.frombuffer(indices_blob[0], dtype=np.int32)
#         dense_fp = np.zeros((1,2048))
#         dense_fp[0,indices]=1
#         fingerprints.append(dense_fp)
    
#     return fingerprints

import pandas as pd
import numpy as np

# def molids_2_fps(cursor=None, mol_ids=None, molecule_df=None, fast=None):
#     fingerprints = []
    
#     if molecule_df is not None:
#         if fast:
#             # Set the DataFrame index to 'zinc_id' for faster lookups
#             molecule_df = molecule_df.set_index('zinc_id', drop=False)
            
#             # Filter the DataFrame for only the relevant Zinc IDs
#             filtered_df = molecule_df.loc[mol_ids]
            
#             # Ensure the DataFrame maintains the order of mol_ids
#             filtered_df = filtered_df.reindex(mol_ids, fill_value=None)
            
#             for mol_id in mol_ids:
#                 row = filtered_df.loc[mol_id]
#                 if pd.notna(row['indices']):
#                     indices = np.frombuffer(row['indices'], dtype=np.int32)
#                     dense_fp = np.zeros((1, 2048))
#                     dense_fp[0, indices] = 1
#                     fingerprints.append(dense_fp)
#                 else:
#                     # Handle missing zinc_id in DataFrame
#                     print(f"Zinc ID {mol_id} not found in the DataFrame.")
#                     fingerprints.append(np.zeros((1, 2048)))  # Or handle appropriately
#         else:
#             # Use the DataFrame to retrieve the fingerprints if provided
#             for mol_id in mol_ids:
#                 row = molecule_df[molecule_df['zinc_id'] == mol_id]
#                 if not row.empty:
#                     indices = np.frombuffer(row['indices'].values[0], dtype=np.int32)
#                     dense_fp = np.zeros((1, 2048))
#                     dense_fp[0, indices] = 1
#                     fingerprints.append(dense_fp)
#                 else:
#                     # Handle missing zinc_id in dataframe
#                     print(f"Zinc ID {mol_id} not found in the dataframe.")
#                     fingerprints.append(np.zeros((1, 2048)))  # Or handle appropriately

#     else:
#         # Use the SQL functionality if the DataFrame is not provided
#         # Use placeholders to query multiple Zinc IDs
#         placeholders = ','.join('?' for _ in mol_ids)
        
#         # Fetch the sparse components for the Zinc IDs
#         cursor.execute(f'''
#             SELECT indices FROM molecules
#             WHERE zinc_id IN ({placeholders})
#         ''', mol_ids)
        
#         results = cursor.fetchall()
#         for indices_blob in results:
#             # Convert binary blobs back to their original arrays
#             indices = np.frombuffer(indices_blob[0], dtype=np.int32)
#             dense_fp = np.zeros((1, 2048))
#             dense_fp[0, indices] = 1
#             fingerprints.append(dense_fp)
    
#     return fingerprints


def molids_2_fps(cursor=None, mol_ids=None, molecule_df=None, fast=None):
    """
    Retrieve fingerprints for given molecule IDs using either a pandas DataFrame or an SQL cursor.
    
    Parameters:
        cursor (sqlite3.Cursor, optional): SQL cursor for database access.
        mol_ids (list): List of molecule IDs (zinc IDs).
        molecule_df (pandas.DataFrame, optional): DataFrame containing molecule information.
        fast (bool, optional): If True, uses a faster batch-based DataFrame lookup.
        
    Returns:
        list: A list of fingerprints (numpy arrays).
    """
    def create_dense_fp(indices):
        """Helper function to create a dense fingerprint from sparse indices."""
        dense_fp = np.zeros((1, 2048))
        dense_fp[0, indices] = 1
        return dense_fp

    fingerprints = []

    if molecule_df is not None:
        if fast:
            # Optimize for large DataFrame: batch retrieval and indexing
            molecule_df = molecule_df.set_index('zinc_id', drop=False)
            # filtered_df = molecule_df.loc[mol_ids].reindex(mol_ids, fill_value=None)

            for mol_id in mol_ids:
                row = molecule_df.loc[mol_id]
                # if pd.notna(row['indices']):
                indices = np.frombuffer(row['indices'], dtype=np.int32)
                fingerprints.append(create_dense_fp(indices))
                # else:
                #     print(f"Zinc ID {mol_id} not found in the DataFrame.")
                #     fingerprints.append(np.zeros((1, 2048)))  # Default empty fingerprint
        else:
            # Sequential lookup: one molecule at a time
            for mol_id in mol_ids:
                row = molecule_df[molecule_df['zinc_id'] == mol_id]
                if not row.empty:
                    indices = np.frombuffer(row['indices'].values[0], dtype=np.int32)
                    fingerprints.append(create_dense_fp(indices))
                else:
                    print(f"Zinc ID {mol_id} not found in the DataFrame.")
                    fingerprints.append(np.zeros((1, 2048)))  # Default empty fingerprint
    elif cursor is not None:
        # Use SQL for retrieving fingerprints
        placeholders = ','.join('?' for _ in mol_ids)
        cursor.execute(f'''
            SELECT indices FROM molecules
            WHERE zinc_id IN ({placeholders})
        ''', mol_ids)

        results = cursor.fetchall()
        for indices_blob in results:
            indices = np.frombuffer(indices_blob[0], dtype=np.int32)
            fingerprints.append(create_dense_fp(indices))
    else:
        raise ValueError("Either a DataFrame or an SQL cursor must be provided.")

    return fingerprints
