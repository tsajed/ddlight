import pickle
import pandas as pd

molecule_df = pickle.load(open("/home/tsajed/phd/ddlight/dataset/Enamine2M_moldf_w_dockscore_5tgt_morgan.pkl",'rb'))

print(molecule_df.head())