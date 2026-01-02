#!/usr/bin/env python3
"""
Analyze WER statistics from CSV files
Provides detailed statistical analysis with reliability metrics
"""

import os
import glob
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict
import json


def load_wer_csv(filepath: str) -> pd.DataFrame:
    """Load WER CSV file"""
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()


def calculate_statistics(wer_values: pd.Series) -> Dict:
    """
    Calculate comprehensive statistics for WER values
    """
    stats = {
        # Basic statistics
        'count': int(len(wer_values)),
        'mean': float(wer_values.mean()),
        'median': float(wer_values.median()),
        'std': float(wer_values.std()),
        'variance': float(wer_values.var()),
        'min': float(wer_values.min()),
        'max': float(wer_values.max()),
        'range': float(wer_values.max() - wer_values.min()),
        
        # Percentiles
        'q1': float(wer_values.quantile(0.25)),
        'q3': float(wer_values.quantile(0.75)),
        'iqr': float(wer_values.quantile(0.75) - wer_values.quantile(0.25)),
        'p5': float(wer_values.quantile(0.05)),
        'p95': float(wer_values.quantile(0.95)),
        
        # Distribution metrics
        'skewness': float(wer_values.skew()),
        'kurtosis': float(wer_values.kurtosis()),
        
        # Reliability metrics
        'cv': float((wer_values.std() / wer_values.mean()) * 100) if wer_values.mean() > 0 else 0,  # Coefficient of Variation
        'sem': float(wer_values.sem()),  # Standard Error of Mean
        'ci_95_lower': float(wer_values.mean() - 1.96 * wer_values.sem()),  # 95% Confidence Interval
        'ci_95_upper': float(wer_values.mean() + 1.96 * wer_values.sem()),
        
        # Performance categories
        'excellent_count': int((wer_values <= 10).sum()),  # WER <= 10%
        'good_count': int(((wer_values > 10) & (wer_values <= 20)).sum()),  # 10% < WER <= 20%
        'fair_count': int(((wer_values > 20) & (wer_values <= 30)).sum()),  # 20% < WER <= 30%
        'poor_count': int((wer_values > 30).sum()),  # WER > 30%
        
        'excellent_pct': float((wer_values <= 10).sum() / len(wer_values) * 100),
        'good_pct': float(((wer_values > 10) & (wer_values <= 20)).sum() / len(wer_values) * 100),
        'fair_pct': float(((wer_values > 20) & (wer_values <= 30)).sum() / len(wer_values) * 100),
        'poor_pct': float((wer_values > 30).sum() / len(wer_values) * 100),
    }
    
    # Round all float values
    for key, value in stats.items():
        if isinstance(value, float):
            stats[key] = round(value, 4)
    
    return stats


def analyze_single_model(csv_file: str) -> Dict:
    """
    Analyze a single model's WER CSV file
    """
    df = load_wer_csv(csv_file)
    
    if df.empty or 'wer' not in df.columns:
        return {}
    
    model_name = Path(csv_file).stem.replace('_wer', '')
    
    stats = calculate_statistics(df['wer'])
    stats['model_name'] = model_name
    stats['csv_file'] = csv_file
    
    return stats


def compare_models(csv_files: List[str]) -> pd.DataFrame:
    """
    Compare statistics across multiple models
    Returns DataFrame with models as rows and metrics as columns
    """
    all_stats = []
    
    for csv_file in csv_files:
        stats = analyze_single_model(csv_file)
        if stats:
            all_stats.append(stats)
    
    if not all_stats:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_stats)
    
    # Reorder columns for better readability
    priority_cols = ['model_name', 'count', 'mean', 'median', 'std', 'min', 'max']
    other_cols = [col for col in df.columns if col not in priority_cols and col != 'csv_file']
    df = df[priority_cols + other_cols + ['csv_file']]
    
    return df


def generate_report(comparison_df: pd.DataFrame, output_dir: str):
    """
    Generate detailed analysis report
    """
    report_file = os.path.join(output_dir, 'wer_analysis_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("WER ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for _, row in comparison_df.iterrows():
            model_name = row['model_name']
            
            f.write(f"\n{'=' * 80}\n")
            f.write(f"MODEL: {model_name}\n")
            f.write(f"{'=' * 80}\n\n")
            
            f.write("BASIC STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Sample Size:        {row['count']}\n")
            f.write(f"  Mean WER:           {row['mean']:.2f}%\n")
            f.write(f"  Median WER:         {row['median']:.2f}%\n")
            f.write(f"  Std Deviation:      {row['std']:.2f}%\n")
            f.write(f"  Min WER:            {row['min']:.2f}%\n")
            f.write(f"  Max WER:            {row['max']:.2f}%\n")
            f.write(f"  Range:              {row['range']:.2f}%\n\n")
            
            f.write("DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Q1 (25th percentile):   {row['q1']:.2f}%\n")
            f.write(f"  Q3 (75th percentile):   {row['q3']:.2f}%\n")
            f.write(f"  IQR:                    {row['iqr']:.2f}%\n")
            f.write(f"  5th percentile:         {row['p5']:.2f}%\n")
            f.write(f"  95th percentile:        {row['p95']:.2f}%\n")
            f.write(f"  Skewness:               {row['skewness']:.4f}\n")
            f.write(f"  Kurtosis:               {row['kurtosis']:.4f}\n\n")
            
            f.write("RELIABILITY METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Coefficient of Variation (CV): {row['cv']:.2f}%\n")
            f.write(f"  Standard Error of Mean (SEM):  {row['sem']:.4f}\n")
            f.write(f"  95% Confidence Interval:       [{row['ci_95_lower']:.2f}%, {row['ci_95_upper']:.2f}%]\n\n")
            
            f.write("PERFORMANCE BREAKDOWN:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Excellent (WER ≤ 10%):  {row['excellent_count']:4d} samples ({row['excellent_pct']:5.1f}%)\n")
            f.write(f"  Good (10% < WER ≤ 20%): {row['good_count']:4d} samples ({row['good_pct']:5.1f}%)\n")
            f.write(f"  Fair (20% < WER ≤ 30%): {row['fair_count']:4d} samples ({row['fair_pct']:5.1f}%)\n")
            f.write(f"  Poor (WER > 30%):       {row['poor_count']:4d} samples ({row['poor_pct']:5.1f}%)\n\n")
            
            f.write("INTERPRETATION:\n")
            f.write("-" * 40 + "\n")
            
            # CV interpretation
            if row['cv'] < 15:
                cv_interp = "Low variability - highly consistent performance"
            elif row['cv'] < 30:
                cv_interp = "Moderate variability - reasonably consistent"
            else:
                cv_interp = "High variability - inconsistent performance"
            f.write(f"  CV: {cv_interp}\n")
            
            # Skewness interpretation
            if abs(row['skewness']) < 0.5:
                skew_interp = "Approximately symmetric distribution"
            elif row['skewness'] > 0:
                skew_interp = "Right-skewed - more samples with low WER"
            else:
                skew_interp = "Left-skewed - more samples with high WER"
            f.write(f"  Skewness: {skew_interp}\n")
            
            # Overall performance
            if row['mean'] <= 10:
                overall = "EXCELLENT"
            elif row['mean'] <= 20:
                overall = "GOOD"
            elif row['mean'] <= 30:
                overall = "FAIR"
            else:
                overall = "POOR"
            f.write(f"  Overall Rating: {overall}\n\n")
        
        # Model comparison
        if len(comparison_df) > 1:
            f.write("\n" + "=" * 80 + "\n")
            f.write("MODEL COMPARISON (Ranked by Mean WER)\n")
            f.write("=" * 80 + "\n\n")
            
            ranked = comparison_df.sort_values('mean')
            for rank, (_, row) in enumerate(ranked.iterrows(), 1):
                f.write(f"{rank}. {row['model_name']:20s} - Mean WER: {row['mean']:6.2f}% (±{row['std']:5.2f}%)\n")
    
    print(f"\nDetailed report saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze WER statistics from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all CSV files in a directory
  python analyze_wer.py -i wer_results
  
  # Analyze specific CSV files
  python analyze_wer.py -f xtts_wer.csv f5tts_wer.csv -o analysis_output
        """
    )
    
    parser.add_argument('-i', '--input-dir',
                        help='Directory containing WER CSV files')
    parser.add_argument('-f', '--files', nargs='+',
                        help='Specific CSV file(s) to analyze')
    parser.add_argument('-o', '--output', default='wer_analysis',
                        help='Output directory (default: wer_analysis)')
    
    args = parser.parse_args()
    
    # Collect CSV files
    csv_files = []
    
    if args.files:
        csv_files = args.files
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            parser.error(f"Input directory not found: {args.input_dir}")
        csv_files = glob.glob(os.path.join(args.input_dir, "*_wer.csv"))
    else:
        parser.error("Must specify either --input-dir or --files")
    
    if not csv_files:
        print("No CSV files found to analyze")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to analyze")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Compare models
    comparison_df = compare_models(csv_files)
    
    if comparison_df.empty:
        print("No valid data to analyze")
        return
    
    # Save comparison CSV
    comparison_file = os.path.join(args.output, 'model_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\nComparison table saved: {comparison_file}")
    
    # Generate detailed report
    generate_report(comparison_df, args.output)
    
    # Save JSON for programmatic access
    json_file = os.path.join(args.output, 'statistics.json')
    stats_dict = comparison_df.to_dict(orient='records')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    print(f"JSON statistics saved: {json_file}")
    
    print(f"\n✓ Analysis complete. Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
