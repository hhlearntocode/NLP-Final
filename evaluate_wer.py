#!/usr/bin/env python3
"""
Evaluate WER (Word Error Rate) between ground-truth and model outputs
Supports multiple model folders for comparison
"""

import os
import glob
import argparse
from pathlib import Path
import pandas as pd
import jiwer
from typing import Dict, List


def load_text_files(folder: str) -> Dict[str, str]:
    """
    Load all .txt files from folder
    Returns: {file_id: text_content}
    """
    txt_files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    data = {}
    
    for filepath in txt_files:
        file_id = Path(filepath).stem 
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        data[file_id] = text
    
    return data


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate WER between reference and hypothesis
    Returns WER as percentage (0-100)
    """
    if not reference or not hypothesis:
        return 100.0 
    
    try:
        wer = jiwer.wer(reference, hypothesis)
        return round(wer * 100, 2)
    except Exception as e:
        print(f"Error calculating WER: {e}")
        return 100.0


def evaluate_single_model(ground_truth_folder: str, model_folder: str, model_name: str) -> pd.DataFrame:
    """
    Evaluate WER for a single model against ground-truth
    Returns DataFrame with columns: id, ground_truth, model_name, wer
    """
    print(f"\n=== Evaluating: {model_name} ===")
    
    # Load data
    ground_truth = load_text_files(ground_truth_folder)
    model_output = load_text_files(model_folder)
    
    # Find common IDs
    common_ids = sorted(set(ground_truth.keys()) & set(model_output.keys()))
    
    if not common_ids:
        print(f"Warning: No common files found between {ground_truth_folder} and {model_folder}")
        return pd.DataFrame()
    
    print(f"Found {len(common_ids)} common files")
    
    # Calculate WER for each file
    results = []
    for file_id in common_ids:
        ref_text = ground_truth[file_id]
        hyp_text = model_output[file_id]
        wer_score = calculate_wer(ref_text, hyp_text)
        
        results.append({
            'id': file_id,
            'ground_truth': ref_text,
            model_name: hyp_text,
            'wer': wer_score
        })
    
    df = pd.DataFrame(results)
    print(f"Average WER: {df['wer'].mean():.2f}%")
    
    return df


def evaluate_multiple_models(ground_truth_folder: str, model_configs: List[Dict[str, str]], output_dir: str):
    """
    Evaluate multiple models and save individual CSV files
    
    model_configs: List of dicts with keys 'name' and 'folder'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for config in model_configs:
        model_name = config['name']
        model_folder = config['folder']
        
        if not os.path.exists(model_folder):
            print(f"Warning: Model folder not found: {model_folder}")
            continue
        
        # Evaluate this model
        df = evaluate_single_model(ground_truth_folder, model_folder, model_name)
        
        if df.empty:
            continue
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"{model_name}_wer.csv")
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Saved: {output_file}")
    
    print(f"\n✓ All evaluations complete. Results saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate WER for TTS model outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model
  python evaluate_wer.py -g ground-truth -m xtts/text -n XTTS -o results
  
  # Multiple models
  python evaluate_wer.py -g ground-truth -m xtts/text f5tts/text -n XTTS F5-TTS -o results
        """
    )
    
    parser.add_argument('-g', '--ground-truth', required=False,default='ground-truth',
                        help='Ground-truth folder path')
    parser.add_argument('-m', '--models', nargs='+', required=True,
                        help='Model output folder(s)')
    parser.add_argument('-n', '--names', nargs='+', required=True,
                        help='Model name(s) (must match number of model folders)')
    parser.add_argument('-o', '--output', default='wer_results',
                        help='Output directory for CSV files (default: wer_results)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.models) != len(args.names):
        parser.error("Number of model folders must match number of model names")
    
    if not os.path.exists(args.ground_truth):
        parser.error(f"Ground-truth folder not found: {args.ground_truth}")
    
    # Build model configs
    model_configs = [
        {'name': name, 'folder': folder}
        for name, folder in zip(args.names, args.models)
    ]
    
    # Run evaluation
    evaluate_multiple_models(args.ground_truth, model_configs, args.output)


if __name__ == "__main__":
    main()
