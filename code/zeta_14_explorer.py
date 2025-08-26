#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zeta_14_explorer.py - Explorer of the connection between the first zero and the number 14
Author: Jefferson M. Okushigue
Date: 2025-08-24
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpmath import mp
import pandas as pd
import pickle
import os
from scipy import stats
from typing import List, Tuple, Dict, Any
import logging

# Configuration
mp.dps = 50
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("plasma")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Zeta14Explorer:
    """Class to explore the connection between the first zero and the number 14"""
    
    def __init__(self, cache_file: str = None):
        self.cache_file = cache_file or self.find_cache_file()
        self.zeros = []
        self.load_zeros()
        
        # First known non-trivial zero
        self.first_nontrivial_zero = 14.134725142
        
        # Constants related to 14
        self.fourteen_related = {
            'first_zero': 14.134725142,
            'exact_14': 14.0,
            'sqrt_196': np.sqrt(196),
            '2_times_7': 2 * 7,
            'e_2_639': np.exp(2.639),
            'pi_2_2': np.pi**2.2,
            'model_parameters': 14,  # Standard Model parameters
            'generations': 3,  # But 14/3 ≈ 4.666...
            'index_ratio': 14.006038  # Ratio found in resonances
        }
    
    def find_cache_file(self):
        """Find the cache file"""
        locations = [
            "zeta_zeros_cache_fundamental.pkl",
            "~/zvt/code/zeta_zeros_cache_fundamental.pkl",
            os.path.expanduser("~/zvt/code/zeta_zeros_cache_fundamental.pkl"),
            "./zeta_zeros_cache_fundamental.pkl"
        ]
        
        for location in locations:
            if os.path.exists(location):
                return location
        return None
    
    def load_zeros(self):
        """Load zeta zeros"""
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.zeros = pickle.load(f)
                logger.info(f"✅ {len(self.zeros):,} zeros loaded")
            except Exception as e:
                logger.error(f"❌ Error loading cache: {e}")
    
    def analyze_first_zero_patterns(self):
        """Analyze patterns related to the first zero"""
        logger.info("🔍 Analyzing patterns of the first non-trivial zero...")
        
        # Compare the first zero with variations of 14
        comparisons = {}
        
        for name, value in self.fourteen_related.items():
            error = abs(self.first_nontrivial_zero - value) / self.first_nontrivial_zero
            comparisons[name] = {
                'value': value,
                'difference': abs(self.first_nontrivial_zero - value),
                'relative_error': error,
                'significance': 'HIGH' if error < 0.01 else 'MEDIUM' if error < 0.1 else 'LOW'
            }
        
        # Create DataFrame for display
        df_data = []
        for name, data in comparisons.items():
            df_data.append({
                'Property': name,
                'Value': data['value'],
                'First Zero': self.first_nontrivial_zero,
                'Difference': data['difference'],
                'Relative Error': f"{data['relative_error']:.2%}",
                'Significance': data['significance']
            })
        
        df = pd.DataFrame(df_data)
        
        print("\n" + "="*80)
        print("ANALYSIS OF THE FIRST NON-TRIVIAL ZERO vs NUMBER 14")
        print("="*80)
        print(df.to_string(index=False))
        
        return comparisons
    
    def find_zeros_starting_with_14(self):
        """Find all zeros that start with 14"""
        if not self.zeros:
            logger.warning("⚠️ No data available")
            return []
        
        logger.info("🔍 Searching for zeros starting with 14...")
        
        zeros_starting_with_14 = []
        
        for idx, gamma in self.zeros:
            gamma_str = f"{gamma:.10f}"
            if gamma_str.startswith("14."):
                zeros_starting_with_14.append((idx, gamma))
        
        logger.info(f"✅ Found {len(zeros_starting_with_14)} zeros starting with 14")
        
        # Display the first 10
        print("\n" + "="*80)
        print("ZEROS STARTING WITH 14 (first 10)")
        print("="*80)
        for i, (idx, gamma) in enumerate(zeros_starting_with_14[:10]):
            print(f"{i+1:2d}. Zero #{idx:8,} → {gamma:.10f}")
        
        if len(zeros_starting_with_14) > 10:
            print(f"... and {len(zeros_starting_with_14) - 10} more zeros")
        
        return zeros_starting_with_14
    
    def analyze_14_in_resonances(self):
        """Analyze how 14 appears in the found resonances"""
        logger.info("🔍 Analyzing the role of 14 in resonances...")
        
        # Known resonances
        resonances = {
            'fine_structure': {
                'index': 118412,
                'gamma': 87144.853030,
                'constant': 1/137.035999084
            },
            'electron_mass': {
                'index': 1658483,
                'gamma': 953397.367271,
                'constant': 9.1093837015e-31
            }
        }
        
        # Calculate relationships with 14
        print("\n" + "="*80)
        print("RELATIONSHIPS OF RESONANCES WITH THE NUMBER 14")
        print("="*80)
        
        for name, data in resonances.items():
            print(f"\n🔬 {name.upper()}:")
            print(f"   Zero index: {data['index']:,}")
            print(f"   Gamma: {data['gamma']:.6f}")
            
            # Direct relationships
            index_div_14 = data['index'] / 14
            gamma_div_14 = data['gamma'] / 14
            
            print(f"   Index / 14 = {index_div_14:.6f}")
            print(f"   Gamma / 14 = {gamma_div_14:.6f}")
            
            # Check if they are close to integers or simple fractions
            index_near_int = round(index_div_14)
            gamma_near_int = round(gamma_div_14)
            
            index_error = abs(index_div_14 - index_near_int) / index_div_14
            gamma_error = abs(gamma_div_14 - gamma_near_int) / gamma_div_14
            
            if index_error < 0.01:
                print(f"   ✅ Index/14 ≈ {index_near_int} (error: {index_error:.2%})")
            
            if gamma_error < 0.01:
                print(f"   ✅ Gamma/14 ≈ {gamma_near_int} (error: {gamma_error:.2%})")
    
    def analyze_14_multiples_pattern(self):
        """Analyze patterns of multiples of 14 in the indices"""
        if not self.zeros:
            return
        
        logger.info("🔍 Analyzing patterns of multiples of 14...")
        
        # Find zeros whose indices are multiples of 14
        multiples_of_14 = []
        
        for idx, gamma in self.zeros:
            if idx % 14 == 0:
                multiples_of_14.append((idx, gamma))
        
        logger.info(f"✅ Found {len(multiples_of_14)} zeros with indices that are multiples of 14")
        
        # Analyze statistics of these zeros
        if multiples_of_14:
            gammas = [g for _, g in multiples_of_14]
            
            print("\n" + "="*80)
            print("STATISTICS OF ZEROS WITH INDICES THAT ARE MULTIPLES OF 14")
            print("="*80)
            print(f"Total: {len(multiples_of_14)} zeros")
            print(f"Mean of γ: {np.mean(gammas):.6f}")
            print(f"Standard deviation: {np.std(gammas):.6f}")
            print(f"Smallest γ: {np.min(gammas):.6f}")
            print(f"Largest γ: {np.max(gammas):.6f}")
            
            # Check if there is a pattern in the differences
            differences = np.diff(gammas)
            print(f"\nMean difference between consecutive γ: {np.mean(differences):.6f}")
            print(f"Standard deviation of differences: {np.std(differences):.6f}")
            
            # Display the first 10
            print(f"\nFirst 10 zeros with indices that are multiples of 14:")
            for i, (idx, gamma) in enumerate(multiples_of_14[:10]):
                print(f"   {i+1:2d}. Zero #{idx:8,} → {gamma:.6f}")
        
        return multiples_of_14
    
    def visualize_14_patterns(self):
        """Create visualizations of patterns related to 14"""
        if not self.zeros:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analysis of Patterns Related to the Number 14', fontsize=16)
        
        # 1. Distribution of first digits
        ax = axes[0, 0]
        first_digits = [int(str(gamma).split('.')[0]) for _, gamma in self.zeros[:10000]]
        
        counts = {}
        for digit in range(10, 20):  # 10 to 19
            counts[digit] = first_digits.count(digit)
        
        bars = ax.bar(counts.keys(), counts.values(), alpha=0.7)
        ax.axvline(x=14, color='red', linestyle='--', label='14')
        ax.set_xlabel('First Digit')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of First Digits (first 10k zeros)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Highlight the bar for 14
        for bar in bars:
            if bar.get_x() == 14:
                bar.set_color('red')
                bar.set_alpha(0.8)
        
        # 2. Zeros starting with 14 vs others
        ax = axes[0, 1]
        zeros_starting_14 = self.find_zeros_starting_with_14()
        
        if zeros_starting_14:
            indices_14 = [idx for idx, _ in zeros_starting_14]
            gammas_14 = [gamma for _, gamma in zeros_starting_14]
            
            # Compare with random zeros
            sample_size = min(len(zeros_starting_14), 1000)
            random_indices = np.random.choice(len(self.zeros), sample_size, replace=False)
            random_gammas = [self.zeros[i][1] for i in random_indices]
            
            ax.scatter(range(len(gammas_14)), gammas_14, alpha=0.6, c='red', label='Start with 14')
            ax.scatter(range(len(random_gammas)), random_gammas, alpha=0.3, c='blue', label='Random sample')
            
            ax.set_xlabel('Index in sample')
            ax.set_ylabel('Value of γ')
            ax.set_title('Zeros starting with 14 vs random sample')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Multiples of 14 in indices
        ax = axes[1, 0]
        multiples_14 = self.analyze_14_multiples_pattern()
        
        if multiples_14:
            indices = [idx for idx, _ in multiples_14[:1000]]  # First 1000
            gammas = [gamma for _, gamma in multiples_14[:1000]]
            
            ax.scatter(indices, gammas, alpha=0.6, c='green')
            ax.set_xlabel('Index (multiple of 14)')
            ax.set_ylabel('Value of γ')
            ax.set_title('Zeros with indices that are multiples of 14')
            ax.grid(True, alpha=0.3)
        
        # 4. Relationship between index and gamma (log scale)
        ax = axes[1, 1]
        sample_indices = [idx for idx, _ in self.zeros[::100]]  # Sparse sample
        sample_gammas = [gamma for _, gamma in self.zeros[::100]]
        
        ax.scatter(sample_indices, sample_gammas, alpha=0.3, s=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Index (log)')
        ax.set_ylabel('γ (log)')
        ax.set_title('Index-γ relationship (log-log scale)')
        ax.grid(True, alpha=0.3)
        
        # Mark known resonances
        resonance_indices = [118412, 1658483]
        resonance_gammas = [87144.853030, 953397.367271]
        ax.scatter(resonance_indices, resonance_gammas, c='red', s=100, 
                  label='Resonances', marker='*')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('zeta_14_patterns.png', dpi=300, bbox_inches='tight')
        logger.info("📊 Visualization saved: zeta_14_patterns.png")
        plt.show()
    
    def run_analysis(self):
        """Run the complete analysis"""
        logger.info("🚀 Starting analysis of the connection with the number 14...")
        
        # 1. Analyze the first zero
        self.analyze_first_zero_patterns()
        
        # 2. Find zeros starting with 14
        self.find_zeros_starting_with_14()
        
        # 3. Analyze 14 in resonances
        self.analyze_14_in_resonances()
        
        # 4. Analyze multiples of 14
        self.analyze_14_multiples_pattern()
        
        # 5. Create visualizations
        self.visualize_14_patterns()
        
        # 6. Conclusions
        print("\n" + "="*80)
        print("CONCLUSIONS OF THE ANALYSIS OF THE NUMBER 14")
        print("="*80)
        print("1. The first non-trivial zero starts with 14.1347...")
        print("2. The ratio between the indices of the resonances is ~14.006")
        print("3. 14 is the number of parameters in the Standard Model")
        print("4. This suggests a possible fundamental connection")
        print("   between the structure of zeta zeros and physics")
        print("5. Further investigations are needed to understand")
        print("   if this is a coincidence or a deep pattern")
        
        logger.info("✅ Analysis completed!")

# Main execution
if __name__ == "__main__":
    try:
        explorer = Zeta14Explorer()
        explorer.run_analysis()
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
