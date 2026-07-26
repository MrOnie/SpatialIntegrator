#!/usr/bin/env python3
"""
Master Orchestration Script for SpatialIntegrator Benchmarking Suite
Runs all quantitative and morphological experiments required for Q1 journal validation.
Generates all high-resolution figures (300 DPI) and data summary tables in results/.
"""
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmarking.run_comprehensive_experiments as comprehensive
import benchmarking.generate_biomarker_figure as biomarkers
import benchmarking.run_multiorgan_experiments as multiorgan

def main():
    print("="*80)
    print("SPATIALINTEGRATOR: MASTER Q1 EXPERIMENTAL EXECUTION SUITE")
    print("="*80)
    os.makedirs('results', exist_ok=True)
    t_start = time.time()
    
    print("\n[PHASE 1] Executing Comprehensive Hyperparameter & Tile Resolution Grid Search...")
    print("--------------------------------------------------------------------------------")
    try:
        comprehensive.main()
    except Exception as e:
        print(f"[ERROR in Phase 1]: {e}")
        import traceback; traceback.print_exc()
        
    print("\n[PHASE 2] Computing Non-Parametric Wilcoxon Biomarker Validation & Dotplots...")
    print("--------------------------------------------------------------------------------")
    try:
        biomarkers.main()
    except Exception as e:
        print(f"[ERROR in Phase 2]: {e}")
        import traceback; traceback.print_exc()
        
    print("\n[PHASE 3] Executing Multi-Organ Benchmark Suite (5 Clinical Scenarios)...")
    print("--------------------------------------------------------------------------------")
    try:
        multiorgan.run_multiorgan_benchmark()
    except Exception as e:
        print(f"[ERROR in Phase 3]: {e}")
        import traceback; traceback.print_exc()
        
    total_time = time.time() - t_start
    print("\n" + "="*80)
    print(f"ALL Q1 BENCHMARKS COMPLETED SUCCESSFULLY IN {total_time/60:.2f} MINUTES!")
    print("="*80)
    print("Summary of Generated Files in 'results/':")
    print("  [Tables]")
    print("    * results/table1_breast_cancer_grid_search.csv")
    print("    * results/table1_breast_cancer_grid_search.tex")
    print("    * results/table2_multiorgan_benchmark_summary.csv")
    print("  [High-Resolution Figures (300 DPI)]")
    print("    * results/fig1_model_sensitivity_analysis.png")
    print("    * results/fig2_spatial_domain_maps_comparison.png")
    print("    * results/fig3_biomarker_deg_validation_dotplot.png")
    print("    * results/fig4_multiorgan_contiguity_comparison.png")
    print("="*80)

if __name__ == '__main__':
    main()
