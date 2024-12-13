import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
from nfp.preprocessing import features
from nfp.preprocessing import Tokenizer
from tqdm import tqdm
import time

DFT_sdf_file_path = '/content/drive/My Drive/CSC461/DFT.sdf'
DFT_df = PandasTools.LoadSDF(DFT_sdf_file_path, smilesName='SMILES', molColName='Molecule', removeHs=False)

stripped_df = DFT_df[['Molecule']]
stripped_df = stripped_df.reset_index(drop=True)
stripped_df['Atom_number'] = np.nan
stripped_df['mol_id'] = np.nan
for i in range(len(stripped_df)):
  stripped_df.at[i, 'Atom_number'] = stripped_df.at[i, 'Molecule'].GetNumAtoms()
  stripped_df.at[i, 'mol_id'] = stripped_df.at[i, 'Molecule'].GetProp('_Name')

stripped_df['Atom_number'] = stripped_df['Atom_number'].astype(int)
stripped_df['mol_id'] = stripped_df['mol_id'].astype(int)

shift_csv_path =  '/content/drive/My Drive/CSC461/DFT8K.csv'
shift_df = pd.read_csv(shift_csv_path, index_col=0)

shift_df = shift_df.loc[shift_df.atom_type == 1]
shift_df['Mol'] = np.nan
id_dft_df = stripped_df.set_index('mol_id')
shift_df = shift_df[shift_df['mol_id'] != 20176318]
shift_df = shift_df.reset_index(drop=True)

for i in range(len(shift_df)):
  mol_id = int(shift_df.at[i, 'mol_id'])
  try:
    shift_df.at[i,'Mol'] = id_dft_df.at[mol_id, 'Molecule']
  except:
    print(mol_id)

group_object = shift_df.groupby(['mol_id'])
shift_list = []
for mol_id,group in group_object:
  shift_list.append([mol_id[0],group.atom_index.values.astype('int'), group.Shift.values.astype('float32')])
  if len(group.atom_index.values) != len(set(group.atom_index.values)):
        print(mol_id)

shift_df_final = pd.DataFrame(shift_list, columns=['mol_id', 'atom_index', 'Shift'])

test = shift_df_final.sample(n = 500, random_state=42)
valid = shift_df_final[~shift_df_final.mol_id.isin(test.mol_id)].sample(n = 500, random_state = 42)
train = shift_df_final[~shift_df_final.mol_id.isin(test.mol_id) & ~shift_df_final.mol_id.isin(valid.mol_id)]

test = test.set_index('mol_id', drop=True)
valid = valid.set_index('mol_id', drop=True)
train = train.set_index('mol_id', drop=True)
final_stripped_df = stripped_df.set_index('mol_id', drop=True)

test = final_stripped_df.reindex(test.index).join(test[['atom_index', 'Shift']])
valid = final_stripped_df.reindex(valid.index).join(valid[['atom_index', 'Shift']])
train = final_stripped_df.reindex(train.index).join(train[['atom_index', 'Shift']])

############################################################################################################################
###############THESE FUNCTIONS ARE TAKEN DIRECTLY FROM CASCADE PAPER########################################################
def atomic_number_tokenizer(atom):
    return atom.GetAtomicNum()
def Mol_iter(df):
    for index,r in df.iterrows():
        yield(r['Mol'], r['atom_index'])

class SmilesPreprocessor(object):
    """ Given a list of SMILES strings, encode these molecules as atom and
    connectivity feature matricies.

    Example:
    >>> preprocessor = SmilesPreprocessor(explicit_hs=False)
    >>> inputs = preprocessor.fit(data.smiles)
    """

    def __init__(self, explicit_hs=True, atom_features=None, bond_features=None):
        """

        explicit_hs : bool
            whether to tell RDkit to add H's to a molecule.
        atom_features : function
            A function applied to an rdkit.Atom that returns some
            representation (i.e., string, integer) for the Tokenizer class.
        bond_features : function
            A function applied to an rdkit Bond to return some description.

        """

        self.atom_tokenizer = Tokenizer()
        self.bond_tokenizer = Tokenizer()
        self.explicit_hs = explicit_hs

        if atom_features is None:
            atom_features = features.atom_features_v1

        if bond_features is None:
            bond_features = features.bond_features_v1

        self.atom_features = atom_features
        self.bond_features = bond_features


    def fit(self, smiles_iterator):
        """ Fit an iterator of SMILES strings, creating new atom and bond
        tokens for unseen molecules. Returns a dictionary with 'atom' and
        'connectivity' entries """
        return list(self.preprocess(smiles_iterator, train=True))


    def predict(self, smiles_iterator):
        """ Uses previously determined atom and bond tokens to convert a SMILES
        iterator into 'atom' and 'connectivity' matrices. Ensures that atom and
        bond classes commute with previously determined results. """
        return list(self.preprocess(smiles_iterator, train=False))


    def preprocess(self, smiles_iterator, train=True):

        self.atom_tokenizer.train = train
        self.bond_tokenizer.train = train

        for smiles in tqdm(smiles_iterator):
            yield self.construct_feature_matrices(smiles)


    @property
    def atom_classes(self):
        """ The number of atom types found (includes the 0 null-atom type) """
        return self.atom_tokenizer.num_classes + 1


    @property
    def bond_classes(self):
        """ The number of bond types found (includes the 0 null-bond type) """
        return self.bond_tokenizer.num_classes + 1


    def construct_feature_matrices(self, smiles):
        """ construct a molecule from the given smiles string and return atom
        and bond classes.

        Returns
        dict with entries
        'n_atom' : number of atoms in the molecule
        'n_bond' : number of bonds in the molecule
        'atom' : (n_atom,) length list of atom classes
        'bond' : (n_bond,) list of bond classes
        'connectivity' : (n_bond, 2) array of source atom, target atom pairs.

        """

        mol = MolFromSmiles(smiles)
        if self.explicit_hs:
            mol = AddHs(mol)

        n_atom = len(mol.GetAtoms())
        n_bond = 2 * len(mol.GetBonds())

        # If its an isolated atom, add a self-link
        if n_bond == 0:
            n_bond = 1

        atom_feature_matrix = np.zeros(n_atom, dtype='int')
        bond_feature_matrix = np.zeros(n_bond, dtype='int')
        connectivity = np.zeros((n_bond, 2), dtype='int')

        bond_index = 0

        atom_seq = mol.GetAtoms()
        atoms = [atom_seq[i] for i in range(n_atom)]

        for n, atom in enumerate(atoms):

            # Atom Classes
            atom_feature_matrix[n] = self.atom_tokenizer(
                self.atom_features(atom))

            start_index = atom.GetIdx()

            for bond in atom.GetBonds():
                # Is the bond pointing at the target atom
                rev = bond.GetBeginAtomIdx() != start_index

                # Bond Classes
                bond_feature_matrix[n] = self.bond_tokenizer(
                    self.bond_features(bond, flipped=rev))

                # Connectivity
                if not rev:  # Original direction
                    connectivity[bond_index, 0] = bond.GetBeginAtomIdx()
                    connectivity[bond_index, 1] = bond.GetEndAtomIdx()

                else:  # Reversed
                    connectivity[bond_index, 0] = bond.GetEndAtomIdx()
                    connectivity[bond_index, 1] = bond.GetBeginAtomIdx()

                bond_index += 1


        return {
            'n_atom': n_atom,
            'n_bond': n_bond,
            'atom': atom_feature_matrix,
            'bond': bond_feature_matrix,
            'connectivity': connectivity,
        }


class ConnectivityAPreprocessor(object):
    """ Given a list of SMILES strings, encode these molecules as atom and
    connectivity feature matricies.

    Example:
    >>> preprocessor = SmilesPreprocessor(explicit_hs=False)
    >>> inputs = preprocessor.fit(data.smiles)
    """

    def __init__(self, explicit_hs=True, atom_features=None, bond_features=None):
        """

        explicit_hs : bool
            whether to tell RDkit to add H's to a molecule.
        atom_features : function
            A function applied to an rdkit.Atom that returns some
            representation (i.e., string, integer) for the Tokenizer class.
        bond_features : function
            A function applied to an rdkit Bond to return some description.

        """

        self.atom_tokenizer = Tokenizer()
        self.bond_tokenizer = Tokenizer()
        self.explicit_hs = explicit_hs

        if atom_features is None:
            atom_features = features.atom_features_v1

        if bond_features is None:
            bond_features = features.bond_features_v1

        self.atom_features = atom_features
        self.bond_features = bond_features


    def fit(self, smiles_iterator):
        """ Fit an iterator of SMILES strings, creating new atom and bond
        tokens for unseen molecules. Returns a dictionary with 'atom' and
        'connectivity' entries """
        return list(self.preprocess(smiles_iterator, train=True))


    def predict(self, smiles_iterator):
        """ Uses previously determined atom and bond tokens to convert a SMILES
        iterator into 'atom' and 'connectivity' matrices. Ensures that atom and
        bond classes commute with previously determined results. """
        return list(self.preprocess(smiles_iterator, train=False))


    def preprocess(self, smiles_iterator, train=True):

        self.atom_tokenizer.train = train
        self.bond_tokenizer.train = train

        for smiles in tqdm(smiles_iterator):
            yield self.construct_feature_matrices(smiles)


    @property
    def atom_classes(self):
        """ The number of atom types found (includes the 0 null-atom type) """
        return self.atom_tokenizer.num_classes + 1


    @property
    def bond_classes(self):
        """ The number of bond types found (includes the 0 null-bond type) """
        return self.bond_tokenizer.num_classes + 1


    def construct_feature_matrices(self, smiles):
        """ construct a molecule from the given smiles string and return atom
        and bond classes.

        Returns
        dict with entries
        'n_atom' : number of atoms in the molecule
        'n_bond' : number of bonds in the molecule
        'atom' : (n_atom,) length list of atom classes
        'bond' : (n_bond,) list of bond classes
        'connectivity' : (n_bond, 2) array of source atom, target atom pairs.

        """

        mol = MolFromSmiles(smiles)
        if self.explicit_hs:
            mol = AddHs(mol)

        n_atom = len(mol.GetAtoms())
        n_bond = 2 * len(mol.GetBonds())

        # If its an isolated atom, add a self-link
        if n_bond == 0:
            n_bond = 1

        atom_feature_matrix = np.zeros(n_atom, dtype='int')
        bond_feature_matrix = np.zeros(n_bond, dtype='int')
        connectivity = np.zeros((n_bond, 2), dtype='int')

        bond_index = 0

        atom_seq = mol.GetAtoms()
        atoms = [atom_seq[i] for i in range(n_atom)]

        for n, atom in enumerate(atoms):

            # Atom Classes
            atom_feature_matrix[n] = self.atom_tokenizer(
                self.atom_features(atom))

            start_index = atom.GetIdx()

            for bond in atom.GetBonds():
                # Is the bond pointing at the target atom
                rev = bond.GetBeginAtomIdx() != start_index

                # Bond Classes
                bond_feature_matrix[n] = self.bond_tokenizer(
                    self.bond_features(bond, flipped=rev))

                # Connectivity
                if not rev:  # Original direction
                    connectivity[bond_index, 0] = bond.GetBeginAtomIdx()
                    connectivity[bond_index, 1] = bond.GetEndAtomIdx()

                else:  # Reversed
                    connectivity[bond_index, 0] = bond.GetEndAtomIdx()
                    connectivity[bond_index, 1] = bond.GetBeginAtomIdx()

                bond_index += 1

        return {
            'n_atom': n_atom,
            'n_bond': n_bond,
            'atom': atom_feature_matrix,
            'bond': bond_feature_matrix,
            'connectivity': connectivity,
        }


class MolPreprocessor(SmilesPreprocessor):
    """ I should refactor this into a base class and separate
    SmilesPreprocessor classes. But the idea is that we only need to redefine
    the `construct_feature_matrices` method to have a working preprocessor that
    handles 3D structures.

    We'll pass an iterator of mol objects instead of SMILES strings this time,
    though.

    """

    def __init__(self, n_neighbors, cutoff, **kwargs):
        """ A preprocessor class that also returns distances between
        neighboring atoms. Adds edges for non-bonded atoms to include a maximum
        of n_neighbors around each atom """

        self.n_neighbors = n_neighbors
        self.cutoff = cutoff
        super(MolPreprocessor, self).__init__(**kwargs)


    def construct_feature_matrices(self, mol):
        """ Given an rdkit mol, return atom feature matrices, bond feature
        matrices, and connectivity matrices.

        Returns
        dict with entries
        'n_atom' : number of atoms in the molecule
        'n_bond' : number of edges (likely n_atom * n_neighbors)
        'atom' : (n_atom,) length list of atom classes
        'bond' : (n_bond,) list of bond classes. 0 for no bond
        'distance' : (n_bond,) list of bond distances
        'connectivity' : (n_bond, 2) array of source atom, target atom pairs.

        """

        n_atom = len(mol.GetAtoms())

        # n_bond is actually the number of atom-atom pairs, so this is defined
        # by the number of neighbors for each atom.
        #if there is cutoff,
        distance_matrix = Chem.Get3DDistanceMatrix(mol)

        if self.n_neighbors <= (n_atom - 1):
            n_bond = self.n_neighbors * n_atom
        else:
            # If there are fewer atoms than n_neighbors, all atoms will be
            # connected
            n_bond = distance_matrix[(distance_matrix < self.cutoff) & (distance_matrix != 0)].size

        if n_bond == 0: n_bond = 1

        # Initialize the matrices to be filled in during the following loop.
        atom_feature_matrix = np.zeros(n_atom, dtype='int')
        bond_feature_matrix = np.zeros(n_bond, dtype='int')
        bond_distance_matrix = np.zeros(n_bond, dtype=np.float32)
        connectivity = np.zeros((n_bond, 2), dtype='int')

        # Hopefully we've filtered out all problem mols by now.
        if mol is None:
            raise RuntimeError("Issue in loading mol")

        # Get a list of the atoms in the molecule.
        atom_seq = mol.GetAtoms()
        atoms = [atom_seq[i] for i in range(n_atom)]

        # Here we loop over each atom, and the inner loop iterates over each
        # neighbor of the current atom.
        bond_index = 0  # keep track of our current bond.
        for n, atom in enumerate(atoms):

            # update atom feature matrix
            atom_feature_matrix[n] = self.atom_tokenizer(
                self.atom_features(atom))

            # if n_neighbors is greater than total atoms, then each atom is a
            # neighbor.
            if (self.n_neighbors + 1) > len(mol.GetAtoms()):
                neighbor_end_index = len(mol.GetAtoms())
            else:
                neighbor_end_index = (self.n_neighbors + 1)

            distance_atom = distance_matrix[n, :]
            cutoff_end_index = distance_atom[distance_atom < self.cutoff].size

            end_index = min(neighbor_end_index, cutoff_end_index)

            # Loop over each of the nearest neighbors

            neighbor_inds = distance_matrix[n, :].argsort()[1:end_index]
            if len(neighbor_inds)==0: neighbor_inds = [n]
            for neighbor in neighbor_inds:

                # update bond feature matrix
                bond = mol.GetBondBetweenAtoms(n, int(neighbor))
                if bond is None:
                    bond_feature_matrix[bond_index] = 0
                else:
                    rev = False if bond.GetBeginAtomIdx() == n else True
                    bond_feature_matrix[bond_index] = self.bond_tokenizer(
                        self.bond_features(bond, flipped=rev))

                distance = distance_matrix[n, neighbor]
                bond_distance_matrix[bond_index] = distance

                # update connectivity matrix
                connectivity[bond_index, 0] = n
                connectivity[bond_index, 1] = neighbor

                bond_index += 1
        print(connectivity)

        return {
            'n_atom': n_atom,
            'n_bond': n_bond,
            'atom': atom_feature_matrix,
            'bond': bond_feature_matrix,
            'distance': bond_distance_matrix,
            'connectivity': connectivity,
        }


class MolBPreprocessor(MolPreprocessor):
    """
    This is a subclass of Molpreprocessor that preprocessor molecule with
    bond property target
    """
    def __init__(self, **kwargs):
        """
        A preprocessor class that also returns bond_target_matrix, besides the bond matrix
        returned by MolPreprocessor. The bond_target_matrix is then used as ref to reduce molecule
        to bond property
        """
        super(MolBPreprocessor, self).__init__(**kwargs)

    def construct_feature_matrices(self, entry):
        """
        Given an entry contining rdkit molecule, bond_index and for the target property,
        return atom
        feature matrices, bond feature matrices, distance matrices, connectivity matrices and bond
        ref matrices.

        returns
        dict with entries
        see MolPreproccessor
        'bond_index' : ref array to the bond index
        """
        mol, bond_index_array = entry

        n_atom = len(mol.GetAtoms())
        n_pro = len(bond_index_array)

        # n_bond is actually the number of atom-atom pairs, so this is defined
        # by the number of neighbors for each atom.
        #if there is cutoff,
        distance_matrix = Chem.Get3DDistanceMatrix(mol)

        if self.n_neighbors <= (n_atom - 1):
            n_bond = self.n_neighbors * n_atom
        else:
            # If there are fewer atoms than n_neighbors, all atoms will be
            # connected
            n_bond = distance_matrix[(distance_matrix < self.cutoff) & (distance_matrix != 0)].size

        if n_bond == 0: n_bond = 1

        # Initialize the matrices to be filled in during the following loop.
        atom_feature_matrix = np.zeros(n_atom, dtype='int')
        bond_feature_matrix = np.zeros(n_bond, dtype='int')
        bond_distance_matrix = np.zeros(n_bond, dtype=np.float32)
        bond_index_matrix = np.full(n_bond, -1, dtype='int')
        connectivity = np.zeros((n_bond, 2), dtype='int')

        # Hopefully we've filtered out all problem mols by now.
        if mol is None:
            raise RuntimeError("Issue in loading mol")

        # Get a list of the atoms in the molecule.
        atom_seq = mol.GetAtoms()
        atoms = [atom_seq[i] for i in range(n_atom)]

        # Here we loop over each atom, and the inner loop iterates over each
        # neighbor of the current atom.
        bond_index = 0  # keep track of our current bond.
        for n, atom in enumerate(atoms):
            # update atom feature matrix
            atom_feature_matrix[n] = self.atom_tokenizer(
                self.atom_features(atom))

            # if n_neighbors is greater than total atoms, then each atom is a
            # neighbor.
            if (self.n_neighbors + 1) > len(mol.GetAtoms()):
                neighbor_end_index = len(mol.GetAtoms())
            else:
                neighbor_end_index = (self.n_neighbors + 1)

            distance_atom = distance_matrix[n, :]
            cutoff_end_index = distance_atom[distance_atom < self.cutoff].size

            end_index = min(neighbor_end_index, cutoff_end_index)

            # Loop over each of the nearest neighbors

            neighbor_inds = distance_matrix[n, :].argsort()[1:end_index]
            if len(neighbor_inds)==0: neighbor_inds = [n]
            for neighbor in neighbor_inds:

                # update bond feature matrix
                bond = mol.GetBondBetweenAtoms(n, int(neighbor))
                if bond is None:
                    bond_feature_matrix[bond_index] = 0
                else:
                    rev = False if bond.GetBeginAtomIdx() == n else True
                    bond_feature_matrix[bond_index] = self.bond_tokenizer(
                        self.bond_features(bond, flipped=rev))
                    try:
                        bond_index_matrix[bond_index] = bond_index_array.tolist().index(bond.GetIdx())
                    except:
                        pass

                distance = distance_matrix[n, neighbor]
                bond_distance_matrix[bond_index] = distance

                # update connectivity matrix
                connectivity[bond_index, 0] = n
                connectivity[bond_index, 1] = neighbor

                bond_index += 1
        return {
            'n_atom': n_atom,
            'n_bond': n_bond,
            'n_pro': n_pro,
            'atom': atom_feature_matrix,
            'bond': bond_feature_matrix,
            'distance': bond_distance_matrix,
            'connectivity': connectivity,
            'bond_index': bond_index_matrix,
        }

class MolAPreprocessor(MolPreprocessor):
    """
    This is a subclass of Molpreprocessor that preprocessor molecule with
    bond property target
    """
    def __init__(self, **kwargs):
        """
        A preprocessor class that also returns bond_target_matrix, besides the bond matrix
        returned by MolPreprocessor. The bond_target_matrix is then used as ref to reduce molecule
        to bond property
        """
        super(MolAPreprocessor, self).__init__(**kwargs)

    def construct_feature_matrices(self, entry):
        """
        Given an entry contining rdkit molecule, bond_index and for the target property,
        return atom
        feature matrices, bond feature matrices, distance matrices, connectivity matrices and bond
        ref matrices.

        returns
        dict with entries
        see MolPreproccessor
        'bond_index' : ref array to the bond index
        """
        mol, atom_index_array = entry

        n_atom = len(mol.GetAtoms())
        n_pro = len(atom_index_array)

        # n_bond is actually the number of atom-atom pairs, so this is defined
        # by the number of neighbors for each atom.
        #if there is cutoff,
        distance_matrix = Chem.Get3DDistanceMatrix(mol)

        #if self.n_neighbors <= (n_atom - 1):
        #    n_bond = self.n_neighbors * n_atom
        #else:
            # If there are fewer atoms than n_neighbors, all atoms will be
            # connected
        n_bond = distance_matrix[(distance_matrix < self.cutoff) & (distance_matrix != 0)].size

        if n_bond == 0: n_bond = 1

        # Initialize the matrices to be filled in during the following loop.
        atom_feature_matrix = np.zeros(n_atom, dtype='int')
        bond_feature_matrix = np.zeros(n_bond, dtype='int')
        bond_distance_matrix = np.zeros(n_bond, dtype=np.float32)
        atom_index_matrix = np.full(n_atom, -1, dtype='int')
        connectivity = np.zeros((n_bond, 2), dtype='int')

        # Hopefully we've filtered out all problem mols by now.
        if mol is None:
            raise RuntimeError("Issue in loading mol")

        # Get a list of the atoms in the molecule.
        atom_seq = mol.GetAtoms()
        atoms = [atom_seq[i] for i in range(n_atom)]

        # Here we loop over each atom, and the inner loop iterates over each
        # neighbor of the current atom.
        bond_index = 0  # keep track of our current bond.
        for n, atom in enumerate(atoms):
            # update atom feature matrix
            atom_feature_matrix[n] = self.atom_tokenizer(
                self.atom_features(atom))
            try:
                atom_index_matrix[n] = atom_index_array.tolist().index(atom.GetIdx())
            except:
                pass
            # if n_neighbors is greater than total atoms, then each atom is a
            # neighbor.
            if (self.n_neighbors + 1) > len(mol.GetAtoms()):
                neighbor_end_index = len(mol.GetAtoms())
            else:
                neighbor_end_index = (self.n_neighbors + 1)

            distance_atom = distance_matrix[n, :]
            cutoff_end_index = distance_atom[distance_atom < self.cutoff].size

            end_index = min(neighbor_end_index, cutoff_end_index)

            # Loop over each of the nearest neighbors

            neighbor_inds = distance_matrix[n, :].argsort()[1:end_index]
            if len(neighbor_inds)==0: neighbor_inds = [n]
            for neighbor in neighbor_inds:

                # update bond feature matrix
                bond = mol.GetBondBetweenAtoms(n, int(neighbor))
                try:
                    if bond is None:
                        bond_feature_matrix[bond_index] = 0
                    else:
                        rev = False if bond.GetBeginAtomIdx() == n else True
                        bond_feature_matrix[bond_index] = self.bond_tokenizer(
                            self.bond_features(bond, flipped=rev))
                except:
                    print('AAAAAAAAAAAAAAA')
                    print(mol.GetProp('_Name'))
                    print(mol.GetProp('ConfId'))

                distance = distance_matrix[n, neighbor]
                bond_distance_matrix[bond_index] = distance

                # update connectivity matrix
                connectivity[bond_index, 0] = n
                connectivity[bond_index, 1] = neighbor

                bond_index += 1
        return {
            'n_atom': n_atom,
            'n_bond': n_bond,
            'n_pro': n_pro,
            'atom': atom_feature_matrix,
            'bond': bond_feature_matrix,
            'distance': bond_distance_matrix,
            'connectivity': connectivity,
            'atom_index': atom_index_matrix,
        }

#############################################################################################################################
#############################################################################################################################

preprocessor = MolAPreprocessor(n_neighbors=100, cutoff=5, atom_features=atomic_number_tokenizer)
train = train.rename(columns = {'Molecule':'Mol'})
valid = valid.rename(columns = {'Molecule':'Mol'})
test = test.rename(columns = {'Molecule':'Mol'})
training_inputs = preprocessor.fit(Mol_iter(train))
valid_inputs = preprocessor.predict(Mol_iter(valid))
test_inputs = preprocessor.predict(Mol_iter(test))

test.to_pickle('/content/drive/MyDrive/CSC461/test.pkl.gz', compression='gzip')
valid.to_pickle('/content/drive/MyDrive/CSC461/valid.pkl.gz', compression='gzip')
train.to_pickle('/content/drive/MyDrive/CSC461/train.pkl.gz', compression='gzip')
import pickle
with open('/content/drive/MyDrive/CSC461/processed_inputs.p', 'wb') as file:
    pickle.dump({
        'inputs_train': training_inputs,
        'inputs_valid': valid_inputs,
        'inputs_test': test_inputs,
        'preprocessor': preprocessor,
    }, file)