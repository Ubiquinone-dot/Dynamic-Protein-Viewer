
import pandas as pdb
import numpy as np
import torch
import util

# for alignment of proteins into same frames
class AffineTransform(object):
    def __init__(self, xyz1, xyz2):
        '''
        Calculates the affine transform to superimpose xyz1 (mobile) onto xyz2 (stationary)
        '''
        assert xyz1.shape == xyz2.shape, 'The input coordinates must have the same shape'
        self.xyz1 = xyz1  # (B, n, 3)
        self.xyz2 = xyz2
        self.B = xyz1.shape[0]
        
    # properties
    @property
    def centroid1(self):
        return self.xyz1.mean(1)
    
    @property
    def centroid2(self):
        return self.xyz2.mean(1)
    
    @property
    def U(self):
        '''Rotation matrix'''
        # center
        xyz1 = self.xyz1 - self.centroid1
        xyz2 = self.xyz2 - self.centroid2

        # Computation of the covariance matrix
        C = torch.matmul(xyz1.permute(0,2,1), xyz2)

        # Compute optimal rotation matrix using SVD
        try:
            V, S, W = torch.svd(C.to(torch.float32))
        except: #incase ill conditioned doesnt work if full of nans #1e-2
            V, S, W = torch.svd(C.to(torch.float32)+1e-6*C.mean()*torch.rand(C.shape, device=C.device))


        # get sign to ensure right-handedness
        d = torch.ones([self.B,3,3], device=xyz1.device)
        d[:,:,-1] = torch.sign(torch.det(V)*torch.det(W)).unsqueeze(1)

        # Rotation matrix U
        U = torch.matmul(d*V, W.permute(0,2,1)) # (B, 3, 3)
        
        if xyz1.dtype == torch.float16: #set ref to half
            U = U.to(torch.float16)

        return U
        
    @property
    def components(self):
        '''get components of the affine transform'''
        return self.centroid1, self.centroid2, self.U
    
    # Bona fide functions
    def apply(self, xyz):
        '''Apply the affine transform to xyz coordinates
        xyz (torch.tensor, (B, n, 3))
        '''
        return (xyz - self.centroid1) @ self.U + self.centroid2


def parse_pdb(filename, **kwargs):
    '''extract xyz coords for all heavy atoms'''
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines, **kwargs)

def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True):
    # indices of residues observed in the structure
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]
    pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = l[21:22], int(l[22:26]), ' '+l[12:16].strip().ljust(3), l[17:20]
        idx = pdb_idx.index((chain,resNo))
        for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
            if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i,idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)
            
    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]
    seq = np.array(seq)[i_unique]

    out = {'xyz':xyz, # cartesian coordinates, [Lx14]
            'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
            'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
            'seq':np.array(seq), # amino acid sequence, [L]
            'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
           }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            if l[:6]=='HETATM' and not (ignore_het_h and l[77]=='H'):
                info_het.append(dict(
                    idx=int(l[7:11]),
                    atom_id=l[12:16],
                    atom_type=l[77],
                    name=l[16:20]
                ))
                xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out['xyz_het'] = np.array(xyz_het)
        out['info_het'] = info_het

    return out


