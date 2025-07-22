"""
Command Line Interface for Meta-Predictor
"""

import argparse
import sys
import json
from .core.predictor import MetaPredictorWrapper


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Predict metabolites using Meta-Predictor transformer models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  meta-predict --smiles "CCO" --meta-predictor-path ./Meta-Predictor
  meta-predict --smiles "CCO" --n-predictions 10 --output metabolites.json
        """
    )
    
    parser.add_argument(
        "--smiles", "-s", 
        required=True,
        help="SMILES string of the parent compound"
    )
    parser.add_argument(
        "--meta-predictor-path", "-p",
        required=True,
        help="Path to Meta-Predictor installation directory"
    )
    parser.add_argument(
        "--n-predictions", "-n", 
        type=int, 
        default=5,
        help="Number of metabolite predictions (default: 5)"
    )
    parser.add_argument(
        "--device", "-d",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for prediction (default: cpu)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (JSON format)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = MetaPredictorWrapper(
            meta_predictor_path=args.meta_predictor_path,
            device=args.device
        )
        
        # Predict metabolites
        result = predictor.predict_single(
            smiles=args.smiles,
            n_predictions=args.n_predictions
        )
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {args.output}")
        else:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()