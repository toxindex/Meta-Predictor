# Meta-Predictor Package

A Python package for metabolite prediction using transformer-based neural machine translation models.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 
- OpenNMT-py
- RDKit

### From source
```bash
git clone <repository-url>
cd Meta-Predictor
pip install -e .
```

### For development
```bash
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from meta_predictor import MetaPredictorWrapper

# Initialize predictor (requires trained models)
predictor = MetaPredictorWrapper(
    meta_predictor_path="/path/to/Meta-Predictor",
    device="cpu"  # or "cuda" for GPU
)

# Predict metabolites for ethanol
result = predictor.predict_single("CCO", n_predictions=5)
print(f"Found {result['num_metabolites']} metabolites")

# Batch prediction
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
results = predictor.predict_batch(smiles_list)
```

### Command Line Interface

```bash
# Basic prediction
meta-predict --smiles "CCO" --meta-predictor-path ./Meta-Predictor

# More predictions with GPU
meta-predict --smiles "CCO" --n-predictions 10 --device cuda

# Save to file
meta-predict --smiles "CCO" --output metabolites.json
```

## Model Architecture

Meta-Predictor uses a two-stage transformer approach:

1. **Site of Metabolism (SoM) Prediction**: Identifies reactive sites
2. **Metabolite Structure Generation**: Generates metabolite structures

### Models Required

The package expects the following model files in your Meta-Predictor directory:

```
Meta-Predictor/
├── model/
│   ├── SoM_identifier/
│   │   ├── model1.pt
│   │   ├── model2.pt
│   │   ├── model3.pt
│   │   └── model4.pt
│   └── metabolite_predictor/
│       ├── model1.pt
│       ├── model2.pt
│       ├── model3.pt
│       ├── model4.pt
│       └── model5.pt
└── onmt/  # OpenNMT framework
```

## API Reference

### MetaPredictorWrapper

Main prediction class with the following methods:

#### `__init__(meta_predictor_path, device='cpu')`
Initialize the predictor with path to Meta-Predictor installation.

#### `predict_single(smiles, n_predictions=5)`
Predict metabolites for a single compound.

**Parameters:**
- `smiles` (str): Input SMILES string
- `n_predictions` (int): Number of predictions to generate

**Returns:**
- Dictionary with prediction results

#### `predict_batch(smiles_list, n_predictions=5)`
Predict metabolites for multiple compounds.

**Parameters:**
- `smiles_list` (List[str]): List of SMILES strings
- `n_predictions` (int): Number of predictions per compound

**Returns:**
- List of dictionaries with prediction results

## Output Format

```python
{
    'parent_smiles': 'CCO',
    'metabolites': [
        {'smiles': 'CC(=O)O', 'score': 1.0},
        {'smiles': 'CCOS(=O)(=O)O', 'score': 0.8},
        # ... more metabolites
    ],
    'num_metabolites': 5,
    'num_invalid': 0,
    'num_unrational': 2
}
```

## Dependencies

- **torch**: PyTorch framework
- **OpenNMT-py**: Neural machine translation toolkit
- **rdkit**: Molecular structure handling
- **pandas**: Data manipulation
- **numpy**: Numerical operations

## Troubleshooting

### Common Issues

1. **"onmt_translate command not found"**
   - Install OpenNMT-py: `pip install OpenNMT-py`

2. **Model files not found**
   - Ensure trained model files are in the correct directories
   - Check that `meta_predictor_path` points to the right location

3. **CUDA out of memory**
   - Use `device="cpu"` instead of `device="cuda"`
   - Reduce batch size for batch predictions

## Citation

If you use Meta-Predictor in your research, please cite:

[Add appropriate citation here]

## License

[Specify license]