"""
Utility functions for Meta-Predictor
"""

import pandas as pd
import numpy as np
from numpy import nan
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import DataStructs
import math
import re
from typing import Set, Dict, List, Tuple


def canonicalise_smile(smi: str) -> str:
    """Canonicalize a SMILES string"""
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonical


def randomise_smile(smi: str) -> str:
    """Generate a random SMILES representation"""
    mol = Chem.MolFromSmiles(smi)
    random = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    return random


def smi_tokenizer(smi: str) -> str:
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def check_smile(smi: str) -> bool:
    """Check if a SMILES string is valid"""
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return False
    else:
        return True


def count_atoms(smiles: str) -> int:
    """Count the number of atoms in a molecule"""
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms()


def get_added_atoms(reactant, product) -> Tuple[Set[str], Dict[str, int]]:
    """
    Returns the set of symbols of the atoms that have been added in the product and 
    a vocabulary which for each symbol gives the number of added atoms
    """
    reactant_atoms = {}
    product_atoms = {}
    
    for atom in reactant.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in reactant_atoms.keys():
            counter = reactant_atoms[symbol]
        else:
            counter = 0
        counter = counter + 1
        reactant_atoms[symbol] = counter
        
    for atom in product.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in product_atoms.keys():
            counter = product_atoms[symbol]
        else:
            counter = 0
        counter = counter + 1
        product_atoms[symbol] = counter
        
    added_atoms_counts = {}
    added_atoms_set = set()
    
    for symbol in product_atoms.keys():
        counter_prod = product_atoms[symbol]
        if not symbol in reactant_atoms.keys():
            counter_reac = 0
            added_atoms_set.add(symbol)
        else:
            counter_reac = reactant_atoms[symbol]
        if counter_prod > counter_reac:
            diff = counter_prod - counter_reac
            added_atoms_counts[symbol] = diff
            
    return added_atoms_set, added_atoms_counts


def check_added_atoms(reactant, product) -> bool:
    """Check if only allowed atoms are added"""
    pass_test = True
    accepted_atoms = set(['C', 'c', 'O', 'o', 'H', 'S', 's', 'P', 'p', 'N', 'n'])
    added_atoms, counts = get_added_atoms(reactant, product)
    for atom in added_atoms:
        if not atom in accepted_atoms:
            pass_test = False
    return pass_test


def process_predictions(predictions: List[str], drug: str, size_diff_thresh: float = 0.25, 
                       filtering: bool = True) -> Tuple[Set[str], bool, int, int]:
    """Process and filter predicted metabolites"""
    invalid = False
    invalid_count = 0
    canonicalised = set()
    
    for pred in predictions:
        if '.' in pred:
            continue
        try:
            canonical = canonicalise_smile(pred)
            canonicalised.add(canonical)
        except:
            invalid_count = invalid_count + 1
            
    processed = set()
    if invalid_count == len(predictions):
        invalid = True
    else:
        drug_mol = Chem.MolFromSmiles(drug)
        initial_atomcounts = drug_mol.GetNumAtoms()
        
        for pred in canonicalised:
            mol = Chem.MolFromSmiles(pred)
            if mol is None:
                continue
            if filtering:
                if check_added_atoms(drug_mol, mol):
                    if mol.GetNumAtoms() > math.ceil(size_diff_thresh * initial_atomcounts):
                        processed.add(pred)
                        
    unrational_count = len(predictions) - len(processed) - invalid_count
    return processed, invalid, invalid_count, unrational_count


def get_similarity(prediction_smiles: List[str], target_smiles: List[str], 
                  drug: str) -> Tuple[List[float], Set[Tuple[str, str, str]]]:
    """Calculate similarities between predictions and targets"""
    similarities = []
    closest_preds = set()
    
    for target in target_smiles:
        max_similarity = -5
        target_mol = Chem.MolFromSmiles(target)
        target_fgp = Chem.RDKFingerprint(target_mol)
        closest = ''
        
        for prediction in prediction_smiles:
            prediction_mol = Chem.MolFromSmiles(prediction)
            prediction_fgp = Chem.RDKFingerprint(prediction_mol)
            sim = DataStructs.FingerprintSimilarity(target_fgp, prediction_fgp)
            if sim > max_similarity:
                max_similarity = sim
                closest = prediction
                
        similarities.append(max_similarity)
        closest_preds.add((drug, target, closest))
        
    return similarities, closest_preds


def countAtoms(smiles: str) -> int:
    """Count atoms in a SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol is None:
        return mol.GetNumAtoms()
    else:
        return 0