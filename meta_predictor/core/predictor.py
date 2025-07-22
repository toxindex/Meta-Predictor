"""
Core Meta-Predictor functionality
"""

import os
import sys
import subprocess
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

from .utils import (
    canonicalise_smile, smi_tokenizer, check_smile, 
    process_predictions
)

logger = logging.getLogger(__name__)


class MetaPredictorWrapper:
    """
    A wrapper class for Meta-Predictor metabolite prediction
    """
    
    def __init__(self, meta_predictor_path: str, device: str = 'cpu'):
        """
        Initialize Meta-Predictor wrapper
        
        Args:
            meta_predictor_path (str): Path to Meta-Predictor directory
            device (str): Device to use ('cpu' or 'cuda')
        """
        self.meta_predictor_path = Path(meta_predictor_path)
        self.device = device
        self.validate_installation()
        
        # Model paths
        self.som_models = [
            self.meta_predictor_path / "model/SoM_identifier/model1.pt",
            self.meta_predictor_path / "model/SoM_identifier/model2.pt", 
            self.meta_predictor_path / "model/SoM_identifier/model3.pt",
            self.meta_predictor_path / "model/SoM_identifier/model4.pt"
        ]
        
        self.metabolite_models = [
            self.meta_predictor_path / "model/metabolite_predictor/model1.pt",
            self.meta_predictor_path / "model/metabolite_predictor/model2.pt",
            self.meta_predictor_path / "model/metabolite_predictor/model3.pt", 
            self.meta_predictor_path / "model/metabolite_predictor/model4.pt",
            self.meta_predictor_path / "model/metabolite_predictor/model5.pt"
        ]
        
    def validate_installation(self):
        """Validate Meta-Predictor installation"""
        # Check for directories with actual names (spaces instead of underscores)
        som_dirs = [
            "model/SoM identifier",  # Actual directory name
            "model/SoM_identifier"   # Fallback name
        ]
        
        metabolite_dirs = [
            "model/metabolite predictor",  # Actual directory name  
            "model/metabolite_predictor"   # Fallback name
        ]
        
        # Find SoM identifier directory
        som_model_dir = None
        for dir_name in som_dirs:
            dir_path = self.meta_predictor_path / dir_name
            if dir_path.exists():
                som_model_dir = dir_path
                break
                
        if som_model_dir is None:
            raise FileNotFoundError(f"SoM identifier directory not found. Tried: {som_dirs}")
            
        # Find metabolite predictor directory  
        metabolite_model_dir = None
        for dir_name in metabolite_dirs:
            dir_path = self.meta_predictor_path / dir_name
            if dir_path.exists():
                metabolite_model_dir = dir_path
                break
                
        if metabolite_model_dir is None:
            raise FileNotFoundError(f"Metabolite predictor directory not found. Tried: {metabolite_dirs}")
            
        # Check for model files
        if not any(som_model_dir.glob("*.pt")):
            raise FileNotFoundError(f"No model files found in {som_model_dir}")
            
        if not any(metabolite_model_dir.glob("*.pt")):
            raise FileNotFoundError(f"No model files found in {metabolite_model_dir}")
            
        # Check for onmt directory
        onmt_dir = self.meta_predictor_path / "onmt"
        if not onmt_dir.exists():
            raise FileNotFoundError(f"OpenNMT directory not found: {onmt_dir}")
            
        # Update model paths with actual directory names
        self.som_model_dir = som_model_dir
        self.metabolite_model_dir = metabolite_model_dir
            
        logger.info("Meta-Predictor installation validated")
        
    def prepare_input(self, smiles_list: List[str]) -> str:
        """Prepare input file for Meta-Predictor"""
        # Create temporary input file
        temp_input = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        
        for i, smiles in enumerate(smiles_list):
            if not check_smile(smiles):
                logger.warning(f"Invalid SMILES: {smiles}")
                continue
                
            canonical_smiles = canonicalise_smile(smiles)
            tokenized_smiles = smi_tokenizer(canonical_smiles)
            temp_input.write(tokenized_smiles + '\n')
            
        temp_input.close()
        return temp_input.name
        
    def run_prediction_script(self, smiles: str, n_predictions: int = 5) -> str:
        """Run Meta-Predictor using shell scripts"""
        try:
            # Create temporary input CSV file
            temp_input_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_input_csv.write(f"compound_1,{smiles}\n")
            temp_input_csv.close()
            
            # Create temporary output directory
            temp_output_dir = tempfile.mkdtemp()
            
            # Prepare tokenized input file
            temp_tokenized = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            temp_tokenized.close()
            
            # Step 1: Prepare input (tokenize SMILES)
            prep_cmd = [
                "python", "prepare_input_file.py",
                "-input_file", temp_input_csv.name,
                "-output_file", temp_tokenized.name
            ]
            
            result = subprocess.run(
                prep_cmd,
                cwd=self.meta_predictor_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Input preparation failed: {result.stderr}")
            
            # Step 2: Run prediction script based on n_predictions
            script_map = {
                1: "predict-top1.sh",
                5: "predict-top5.sh", 
                10: "predict-top10.sh",
                15: "predict-top15.sh"
            }
            
            # Use closest available script
            script_name = script_map.get(n_predictions, "predict-top5.sh")
            script_path = self.meta_predictor_path / script_name
            
            if not script_path.exists():
                raise FileNotFoundError(f"Prediction script not found: {script_path}")
            
            # Run prediction script
            pred_cmd = [
                "bash", str(script_path),
                temp_tokenized.name,  # SRC_FILE
                temp_output_dir,      # OUT_PATH  
                temp_input_csv.name   # INPUT_FILE
            ]
            
            result = subprocess.run(
                pred_cmd,
                cwd=self.meta_predictor_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.warning(f"Prediction script warning: {result.stderr}")
                # Don't fail on warnings, continue to check output
            
            # Read results
            output_csv = Path(temp_output_dir) / "predict.csv"
            if output_csv.exists():
                return str(output_csv)
            else:
                raise RuntimeError("Prediction output file not found")
                
        except Exception as e:
            logger.error(f"Prediction script failed: {e}")
            raise
            
    def predict_metabolites(self, smiles: str, n_predictions: int = 5) -> Dict:
        """
        Predict metabolites for a single SMILES string using shell scripts
        
        Args:
            smiles (str): Input SMILES string
            n_predictions (int): Number of predictions to generate
            
        Returns:
            Dict: Prediction results
        """
        try:
            logger.info(f"Predicting metabolites for: {smiles}")
            
            # Run prediction using shell scripts
            output_csv_path = self.run_prediction_script(smiles, n_predictions)
            
            # Parse CSV results
            import pandas as pd
            try:
                df = pd.read_csv(output_csv_path)
                if len(df) > 0:
                    predictions_str = df.iloc[0]['predictions'] if 'predictions' in df.columns else ''
                    if predictions_str and isinstance(predictions_str, str):
                        metabolites = predictions_str.split(' ')
                        metabolites = [met.strip() for met in metabolites if met.strip()]
                    else:
                        metabolites = []
                else:
                    metabolites = []
                    
                # Clean up temporary file
                os.unlink(output_csv_path)
                
                return {
                    'parent_smiles': smiles,
                    'metabolites': [{'smiles': met, 'score': 1.0} for met in metabolites],
                    'num_metabolites': len(metabolites),
                    'num_invalid': 0,
                    'num_unrational': 0
                }
                
            except Exception as e:
                logger.error(f"Error parsing CSV results: {e}")
                return {
                    'parent_smiles': smiles,
                    'metabolites': [],
                    'num_metabolites': 0,
                    'error': f"CSV parsing error: {e}"
                }
            
        except Exception as e:
            logger.error(f"Prediction failed for {smiles}: {e}")
            return {
                'parent_smiles': smiles,
                'metabolites': [],
                'num_metabolites': 0,
                'error': str(e)
            }
            
    def predict_batch(self, smiles_list: List[str], n_predictions: int = 5) -> List[Dict]:
        """
        Predict metabolites for a batch of SMILES strings
        
        Args:
            smiles_list (List[str]): List of SMILES strings
            n_predictions (int): Number of predictions per compound
            
        Returns:
            List[Dict]: List of prediction results
        """
        results = []
        for smiles in smiles_list:
            result = self.predict_metabolites(smiles, n_predictions)
            results.append(result)
        return results
        
    def predict_single(self, smiles: str, n_predictions: int = 5) -> Dict:
        """
        Predict metabolites for a single compound (alias for predict_metabolites)
        
        Args:
            smiles (str): Input SMILES string
            n_predictions (int): Number of predictions
            
        Returns:
            Dict: Prediction results
        """
        return self.predict_metabolites(smiles, n_predictions)