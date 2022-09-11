# parses raw (static) pdb files to correct format
# dynamic commiuinication for later
import parsing_utils
import glob

import os, sys, json
import pandas as pd

def main():
	parse_dir = 'raw_data/'
	# grab coordinates by some means
	# select from resid 116 onwwards for larger protein scaffold
	# or select until resid 56 for two helices
	for pdb_file in glob.glob(parse_dir+'*.pdb'):
		data = parsing_utils.parse_pdb(pdb_file)
		dct={
			k:[] for k in ['x','y','z','aa']
		}
		for xyz, aa in zip(data['xyz'], data['seq']):
			CA_coord = xyz[1]  # see util.py, 14HA repr
			dct['x'].append(CA_coord[0])
			dct['y'].append(CA_coord[1])
			dct['z'].append(CA_coord[2])
			dct['aa'].append(aa)
		df = pd.DataFrame(dct)
		outpath = f'parsed_data/{os.path.basename(pdb_file)}_parsed.csv'
		df.to_csv(outpath, index=False)
		print('parsed data to:', outpath)

if __name__=='__main__':
	main()