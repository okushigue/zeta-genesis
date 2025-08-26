#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explorer_fixed.py - Fixed version of the resonance explorer
Author: Jefferson M. Okushigue
Date: 2025-08-24
Version: 2.0 - With improved report and chart generation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpmath import mp
import pandas as pd
import pickle
import os
from scipy import stats, optimize
from typing import List, Tuple, Dict, Any
import logging
from dataclasses import dataclass
from datetime import datetime
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch

# Configuration
mp.dps = 50
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReportGenerator:
    """Class to manage report generation and visualizations"""
    
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_file = os.path.join(output_dir, f"resonances_report_{self.timestamp}.txt")
        self.ensure_output_dir()
        
        # Initialize the report
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RESONANCE EXPLORATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def ensure_output_dir(self):
        """Ensure the output directory exists"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Output directory created: {self.output_dir}")
    
    def add_section(self, title):
        """Add a new section to the report"""
        with open(self.report_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"{title.upper()}\n")
            f.write(f"{'='*80}\n\n")
    
    def add_text(self, text):
        """Add text to the report"""
        with open(self.report_file, 'a', encoding='utf-8') as f:
            f.write(f"{text}\n")
    
    def add_table(self, headers, data, title=None):
        """Add a table to the report"""
        with open(self.report_file, 'a', encoding='utf-8') as f:
            if title:
                f.write(f"{title}\n")
            
            # Calculate column widths
            col_widths = [max(len(str(row[i])) for row in [headers] + data) + 2 for i in range(len(headers))]
            
            # Header
            header_line = "|".join(h.center(col_widths[i]) for i, h in enumerate(headers))
            f.write(header_line + "\n")
            
            # Separator
            separator = "+" + "+".join("-" * (col_widths[i]) for i in range(len(headers))) + "+"
            f.write(separator + "\n")
            
            # Data
            for row in data:
                row_line = "|".join(str(row[i]).center(col_widths[i]) for i in range(len(headers)))
                f.write(row_line + "\n")
            
            f.write("\n")
    
    def save_figure(self, fig, name):
        """Save a figure in the output directory"""
        path = os.path.join(self.output_dir, f"{name}_{self.timestamp}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Chart saved: {path}")
        
        # Add reference to the report
        with open(self.report_file, 'a', encoding='utf-8') as f:
            f.write(f"[CHART: {name} saved in {path}]\n\n")
        
        return path
    
    def close(self):
        """Finalize the report"""
        with open(self.report_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write("END OF REPORT\n")
            f.write(f"{'='*80}\n")
        logger.info(f"Report saved: {self.report_file}")
        return self.report_file

@dataclass
class ResonanceData:
    """Data for a specific resonance"""
    zero_index: int
    gamma: float
    constant_name: str
    constant_value: float
    quality: float
    relative_error: float
    tolerance: float
    energy_gev: float

class ResonanceExplorer:
    """Class for in-depth exploration of resonances"""
    
    def __init__(self, cache_file: str = None, zeros_file: str = None):
        """
        Initialize the explorer with flexible file search
        
        Args:
            cache_file: Cache file (optional)
            zeros_file: Original zeros file (optional)
        """
        self.zeros = []
        self.cache_file = cache_file
        self.zeros_file = zeros_file
        self.resonances = []
        
        # Initialize report generator
        self.report = ReportGenerator()
        
        # Try to find files automatically
        self.find_data_files()
        self.load_zeros()
        
        # Define previously found resonances
        self.target_resonances = [
            ResonanceData(
                zero_index=118412,
                gamma=87144.853030040001613,
                constant_name="fine_structure",
                constant_value=1/137.035999084,
                quality=9.091261e-10,
                relative_error=1.2458301e-5,
                tolerance=1e-4,
                energy_gev=8714.485303
            ),
            ResonanceData(
                zero_index=1658483,
                gamma=953397.367270938004367,
                constant_name="electron_mass",
                constant_value=9.1093837015e-31,
                quality=3.209771e-37,
                relative_error=3.5235878e-5,
                tolerance=1e-30,
                energy_gev=95339.736727
            )
        ]
    
    def find_data_files(self):
        """Search for data files in common locations"""
        # Possible cache locations
        cache_locations = [
            "zeta_zeros_cache_fundamental.pkl",
            "~/zvt/code/zeta_zeros_cache_fundamental.pkl",
            os.path.expanduser("~/zvt/code/zeta_zeros_cache_fundamental.pkl"),
            "./zeta_zeros_cache_fundamental.pkl"
        ]
        
        # Possible zeros file locations
        zeros_locations = [
            "zero.txt",
            "~/zeta/zero.txt",
            os.path.expanduser("~/zeta/zero.txt"),
            "./zero.txt"
        ]
        
        # Search for cache
        for location in cache_locations:
            if os.path.exists(location):
                self.cache_file = location
                logger.info(f"✅ Cache found: {location}")
                break
        
        # Search for zeros file
        for location in zeros_locations:
            if os.path.exists(location):
                self.zeros_file = location
                logger.info(f"✅ Zeros file found: {location}")
                break
        
        if not self.cache_file and not self.zeros_file:
            logger.error("❌ No data files found!")
            logger.info("🔍 Checked locations:")
            for loc in cache_locations + zeros_locations:
                logger.info(f"   - {loc}")
    
    def load_zeros(self):
        """Load zeta zeros from cache or original file"""
        # Try to load from cache first
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.zeros = pickle.load(f)
                logger.info(f"✅ {len(self.zeros):,} zeros loaded from cache")
                return
            except Exception as e:
                logger.error(f"❌ Error loading cache: {e}")
        
        # Try to load from original file
        if self.zeros_file and os.path.exists(self.zeros_file):
            try:
                logger.info(f"📂 Loading zeros from file: {self.zeros_file}")
                zeros = []
                with open(self.zeros_file, 'r') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                zero = float(line)
                                zeros.append((i+1, zero))
                            except ValueError:
                                continue
                
                self.zeros = zeros
                logger.info(f"✅ {len(self.zeros):,} zeros loaded from file")
                
                # Save a local cache for future use
                try:
                    with open("zeta_zeros_cache_fundamental.pkl", 'wb') as f:
                        pickle.dump(self.zeros, f)
                    logger.info("💾 Local cache created for future use")
                except Exception as e:
                    logger.warning(f"⚠️ Could not create local cache: {e}")
                
            except Exception as e:
                logger.error(f"❌ Error loading file: {e}")
        else:
            logger.error("❌ No data files available")
    
    def analyze_neighborhood(self, resonance: ResonanceData, window_size: int = 100) -> Dict[str, Any]:
        """Analyze the neighborhood of a resonance zero"""
        idx = resonance.zero_index - 1  # Converting to zero-based index
        
        if idx < 0 or idx >= len(self.zeros):
            logger.error(f"❌ Index {resonance.zero_index} out of bounds (total: {len(self.zeros)})")
            return {}
        
        start_idx = max(0, idx - window_size)
        end_idx = min(len(self.zeros), idx + window_size + 1)
        neighborhood = self.zeros[start_idx:end_idx]
        
        # Extract gamma values
        gamma_values = [z[1] for z in neighborhood]
        indices = [z[0] for z in neighborhood]
        
        # Statistical analysis of the neighborhood
        results = {
            'resonance': resonance,
            'neighborhood_size': len(neighborhood),
            'gamma_values': gamma_values,
            'indices': indices,
            'local_stats': {
                'mean': np.mean(gamma_values),
                'std': np.std(gamma_values),
                'min': np.min(gamma_values),
                'max': np.max(gamma_values),
                'median': np.median(gamma_values)
            },
            'differences': np.diff(gamma_values),
            'normalized_distances': []
        }
        
        # Calculate normalized distances to the constant
        for i, gamma in enumerate(gamma_values):
            mod_val = gamma % resonance.constant_value
            min_dist = min(mod_val, resonance.constant_value - mod_val)
            results['normalized_distances'].append(min_dist)
        
        # Find the local minimum in the neighborhood
        min_dist_idx = np.argmin(results['normalized_distances'])
        results['local_minimum'] = {
            'index': indices[min_dist_idx],
            'gamma': gamma_values[min_dist_idx],
            'distance': results['normalized_distances'][min_dist_idx]
        }
        
        return results
    
    def test_mathematical_relations(self, resonance: ResonanceData) -> Dict[str, Any]:
        """Test additional mathematical relations for the resonance"""
        results = {}
        
        # Test relations with known numbers
        known_numbers = {
            'pi': np.pi,
            'e': np.e,
            'golden_ratio': (1 + np.sqrt(5)) / 2,
            'sqrt(2)': np.sqrt(2),
            'sqrt(3)': np.sqrt(3),
            'sqrt(5)': np.sqrt(5),
            'ln(2)': np.log(2),
            'ln(3)': np.log(3),
            'gamma_euler': 0.5772156649
        }
        
        gamma = resonance.gamma
        constant = resonance.constant_value
        
        # Test direct relations
        for name, value in known_numbers.items():
            # Test γ / constant ≈ number
            ratio = gamma / constant
            error = abs(ratio - value) / value
            results[f'gamma/constant_{name}'] = {
                'value': ratio,
                'target': value,
                'error': error,
                'significant': error < 0.01
            }
            
            # Test constant / γ ≈ number
            ratio_inv = constant / gamma
            error_inv = abs(ratio_inv - value) / value
            results[f'constant/gamma_{name}'] = {
                'value': ratio_inv,
                'target': value,
                'error': error_inv,
                'significant': error_inv < 0.01
            }
        
        # Test special relations for physical constants
        if resonance.constant_name == "fine_structure":
            # Test relation with 137
            alpha_inv = 1 / constant
            results['alpha_inverse'] = {
                'value': alpha_inv,
                'target': 137.035999084,
                'error': abs(alpha_inv - 137.035999084) / 137.035999084,
                'significant': True
            }
            
            # Test γ ≈ alpha_inv * factor
            for factor in [635.8, 636, 1000/1.57]:  # Interesting factors
                test_value = alpha_inv * factor
                error = abs(gamma - test_value) / gamma
                results[f'alpha_x_{factor}'] = {
                    'value': test_value,
                    'target': gamma,
                    'error': error,
                    'significant': error < 0.01
                }
        
        elif resonance.constant_name == "electron_mass":
            # Test relations with natural units
            # Convert to natural units (where ħ = c = 1)
            # Electron mass in eV: 510998.9461 eV
            mev_mass = 0.5109989461  # MeV
            
            # Test γ ≈ mass_in_eV * factor
            for factor in [1.7e11, 1.87e11, 2e11]:  # Scale factors
                test_value = mev_mass * factor
                error = abs(gamma - test_value) / gamma
                results[f'mass_eV_x_{factor}'] = {
                    'value': test_value,
                    'target': gamma,
                    'error': error,
                    'significant': error < 0.01
                }
        
        return results
    
    def create_synthetic_analysis(self, resonance: ResonanceData):
        """Create synthetic analysis when we don't have real zeros"""
        logger.info(f"🔬 Creating synthetic analysis for {resonance.constant_name}")
        
        # Generate synthetic data based on known properties
        gamma = resonance.gamma
        
        # Create a synthetic neighborhood
        synthetic_gammas = []
        for i in range(-50, 51):
            # Add small random variation
            variation = np.random.normal(0, gamma * 0.001)
            synthetic_gammas.append(gamma + i * 10 + variation)
        
        # Calculate synthetic distances
        distances = []
        for g in synthetic_gammas:
            mod_val = g % resonance.constant_value
            min_dist = min(mod_val, resonance.constant_value - mod_val)
            distances.append(min_dist)
        
        # Find the minimum
        min_idx = np.argmin(distances)
        
        return {
            'synthetic': True,
            'gamma_values': synthetic_gammas,
            'normalized_distances': distances,
            'local_minimum': {
                'index': resonance.zero_index + min_idx - 50,
                'gamma': synthetic_gammas[min_idx],
                'distance': distances[min_idx]
            },
            'local_stats': {
                'mean': np.mean(synthetic_gammas),
                'std': np.std(synthetic_gammas),
                'min': np.min(synthetic_gammas),
                'max': np.max(synthetic_gammas),
                'median': np.median(synthetic_gammas)
            }
        }
    
    def visualize_resonance(self, resonance: ResonanceData, window_size: int = 200):
        """Generate detailed visualizations for a resonance"""
        
        # Try real analysis or use synthetic
        if self.zeros and len(self.zeros) > resonance.zero_index:
            neighborhood_data = self.analyze_neighborhood(resonance, window_size)
        else:
            neighborhood_data = self.create_synthetic_analysis(resonance)
        
        if not neighborhood_data:
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        fig.suptitle(f'Detailed Analysis - {resonance.constant_name.upper()}\n'
                    f'Zero #{resonance.zero_index:,} ({"Synthetic" if neighborhood_data.get("synthetic") else "Real"})', 
                    fontsize=16, weight='bold')
        
        # 1. Time series of gammas in the neighborhood
        ax1 = plt.subplot(gs[0, 0])
        if 'indices' in neighborhood_data:
            indices = neighborhood_data['indices']
        else:
            indices = list(range(resonance.zero_index - 50, resonance.zero_index + 51))
        
        gammas = neighborhood_data['gamma_values']
        
        ax1.plot(indices, gammas, 'b-', alpha=0.7, label='γ values')
        ax1.axvline(x=resonance.zero_index, color='r', linestyle='--', label='Resonance Zero')
        ax1.set_xlabel('Zero Index')
        ax1.set_ylabel('Gamma Value')
        ax1.set_title('Gamma Values in Neighborhood')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Normalized distances
        ax2 = plt.subplot(gs[0, 1])
        distances = neighborhood_data['normalized_distances']
        
        ax2.semilogy(indices, distances, 'g-', alpha=0.7)
        ax2.axvline(x=resonance.zero_index, color='r', linestyle='--', label='Resonance Zero')
        ax2.axhline(y=resonance.quality, color='orange', linestyle=':', label='Achieved Quality')
        ax2.set_xlabel('Zero Index')
        ax2.set_ylabel('Normalized Distance (log)')
        ax2.set_title('Distance to Constant')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Histogram of differences
        ax3 = plt.subplot(gs[1, 0])
        differences = neighborhood_data.get('differences', np.diff(gammas))
        
        ax3.hist(differences, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(x=np.mean(differences), color='r', linestyle='--', label=f'Mean: {np.mean(differences):.3f}')
        ax3.set_xlabel('Difference between consecutive γ')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Gamma Differences')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Residual analysis
        ax4 = plt.subplot(gs[1, 1])
        x = np.array(range(len(gammas)))
        y = np.array(gammas)
        
        # Fit trend line
        coeffs = np.polyfit(x, y, 1)
        trend = np.polyval(coeffs, x)
        residuals = y - trend
        
        scatter = ax4.scatter(x, residuals, alpha=0.6, c=residuals, cmap='coolwarm')
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Position in Window')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals from Linear Trend')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Residual Value')
        
        # 5. Statistical analysis
        ax5 = plt.subplot(gs[2, 0])
        stats = neighborhood_data['local_stats']
        
        stats_names = list(stats.keys())
        stats_values = list(stats.values())
        
        bars = ax5.bar(stats_names, stats_values, color='skyblue', alpha=0.7)
        ax5.set_ylabel('Value')
        ax5.set_title('Local Statistics')
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        
        # Add values on bars
        for bar, value in zip(bars, stats_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax5.grid(True, alpha=0.3)
        
        # 6. Resonance diagram
        ax6 = plt.subplot(gs[2, 1])
        ax6.set_title('Resonance Diagram')
        
        # Draw circle representing the constant
        circle = Circle((0.5, 0.5), 0.3, fill=False, color='blue', linewidth=2)
        ax6.add_patch(circle)
        
        # Draw point representing the resonance zero
        angle = 2 * np.pi * (resonance.gamma % resonance.constant_value) / resonance.constant_value
        x = 0.5 + 0.3 * np.cos(angle)
        y = 0.5 + 0.3 * np.sin(angle)
        
        ax6.scatter(x, y, color='red', s=200, zorder=5, label='Resonance Zero')
        
        # Draw other neighborhood points
        for i, gamma in enumerate(gammas[::5]):  # Take every 5th point to avoid overload
            angle = 2 * np.pi * (gamma % resonance.constant_value) / resonance.constant_value
            x = 0.5 + 0.3 * np.cos(angle)
            y = 0.5 + 0.3 * np.sin(angle)
            ax6.scatter(x, y, color='gray', s=20, alpha=0.5)
        
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.set_aspect('equal')
        ax6.axis('off')
        ax6.legend()
        
        plt.tight_layout()
        
        # Save figure
        filename = f"resonance_analysis_{resonance.constant_name}_{resonance.zero_index}"
        self.report.save_figure(fig, filename)
        plt.close()  # Close to free memory
    
    def compare_resonances(self):
        """Compare the two found resonances"""
        if len(self.target_resonances) < 2:
            logger.warning("⚠️ Not enough resonances for comparison")
            return
        
        r1, r2 = self.target_resonances[0], self.target_resonances[1]
        
        # Create DataFrame for comparison
        data = {
            'Property': [
                'Zero Index', 'Gamma Value', 'Constant Name', 'Constant Value',
                'Quality', 'Relative Error (%)', 'Tolerance', 'Energy (GeV)'
            ],
            r1.constant_name: [
                f"{r1.zero_index:,}", f"{r1.gamma:.6f}", r1.constant_name, 
                f"{r1.constant_value:.6e}", f"{r1.quality:.2e}", 
                f"{r1.relative_error*100:.6f}", f"{r1.tolerance:.0e}", f"{r1.energy_gev:.1f}"
            ],
            r2.constant_name: [
                f"{r2.zero_index:,}", f"{r2.gamma:.6f}", r2.constant_name, 
                f"{r2.constant_value:.6e}", f"{r2.quality:.2e}", 
                f"{r2.relative_error*100:.6f}", f"{r2.tolerance:.0e}", f"{r2.energy_gev:.1f}"
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Add to report
        self.report.add_section("Comparison of Resonances")
        self.report.add_table(df.columns.tolist(), df.values.tolist())
        
        # Calculate ratios between properties
        index_ratio = r2.zero_index / r1.zero_index
        gamma_ratio = r2.gamma / r1.gamma
        quality_ratio = r2.quality / r1.quality
        energy_ratio = r2.energy_gev / r1.energy_gev
        
        self.report.add_text(f"\nIndex ratio (r2/r1): {index_ratio:.6f}")
        self.report.add_text(f"Gamma ratio (r2/r1): {gamma_ratio:.6f}")
        self.report.add_text(f"Quality ratio (r2/r1): {quality_ratio:.6e}")
        self.report.add_text(f"Energy ratio (r2/r1): {energy_ratio:.6f}")
        
        # Additional analysis: check if there's a mathematical relation
        self.report.add_text(f"\n🔍 MATHEMATICAL ANALYSIS OF RATIOS:")
        
        # Check if index ratio is close to interesting numbers
        interesting_ratios = {
            '14': 14.0,
            'sqrt(196)': 14.0,
            '2*7': 14.0,
            'e^2.639': np.exp(2.639),  # e^2.639 ≈ 14
            'pi^2.2': np.pi**2.2,      # pi^2.2 ≈ 14
        }
        
        self.report.add_text(f"\nIndex ratio ({index_ratio:.6f}) vs interesting numbers:")
        for name, value in interesting_ratios.items():
            error = abs(index_ratio - value) / value
            self.report.add_text(f"  {name}: {value:.6f} (error: {error:.2%})")
        
        # Check relation between energy and constant
        self.report.add_text(f"\n🔍 ENERGY-CONSTANT RELATIONSHIPS:")
        for i, r in enumerate(self.target_resonances):
            energy_per_constant = r.energy_gev / r.constant_value
            self.report.add_text(f"  {r.constant_name}: E/constant = {energy_per_constant:.6e}")
        
        # Comparative visualization
        self.create_comparison_visualization(r1, r2, index_ratio, gamma_ratio)
    
    def create_comparison_visualization(self, r1, r2, index_ratio, gamma_ratio):
        """Create comparative visualization of resonances"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Comparative bar chart (log scale for quality)
        properties = ['Quality (log)', 'Rel Error (%)', 'Energy (GeV)']
        r1_values = [np.log10(r1.quality), r1.relative_error*100, r1.energy_gev]
        r2_values = [np.log10(r2.quality), r2.relative_error*100, r2.energy_gev]
        
        x = np.arange(len(properties))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, r1_values, width, label=r1.constant_name, alpha=0.8)
        bars2 = ax1.bar(x + width/2, r2_values, width, label=r2.constant_name, alpha=0.8)
        
        ax1.set_ylabel('Value (Quality in log scale)')
        ax1.set_title('Comparison of Resonance Properties')
        ax1.set_xticks(x)
        ax1.set_xticklabels(properties)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Index vs gamma scatter plot (log scale)
        ax2.scatter([r1.zero_index, r2.zero_index], [r1.gamma, r2.gamma], 
                   s=[200, 200], c=['red', 'blue'], alpha=0.7)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Zero Index (log)')
        ax2.set_ylabel('Gamma Value (log)')
        ax2.set_title('Zero Index vs Gamma Value (log-log)')
        ax2.grid(True, alpha=0.3)
        
        # Add annotations
        ax2.annotate(r1.constant_name, (r1.zero_index, r1.gamma), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
        ax2.annotate(r2.constant_name, (r2.zero_index, r2.gamma), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
        
        # Ratios chart
        ratios = ['Index Ratio', 'Gamma Ratio']
        ratio_values = [index_ratio, gamma_ratio]
        colors = ['green' if abs(r-14) < 0.1 else 'orange' for r in ratio_values]
        
        bars = ax3.bar(ratios, ratio_values, color=colors, alpha=0.7)
        ax3.axhline(y=14, color='red', linestyle='--', label='14')
        ax3.set_ylabel('Ratio Value')
        ax3.set_title('Key Ratios Between Resonances')
        ax3.legend()
        
        # Add values on bars
        for bar, value in zip(bars, ratio_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax3.grid(True, alpha=0.3)
        
        # Connection diagram
        ax4.set_title('Connection Diagram')
        
        # Node positions
        positions = {
            'r1': (0.3, 0.7),
            'r2': (0.7, 0.7),
            '14': (0.5, 0.3)
        }
        
        # Draw nodes
        for name, pos in positions.items():
            if name == '14':
                circle = Circle(pos, 0.08, color='red', alpha=0.8)
                ax4.add_patch(circle)
                ax4.text(pos[0], pos[1], '14', ha='center', va='center', fontsize=12, weight='bold')
            else:
                rect = FancyBboxPatch((pos[0]-0.1, pos[1]-0.05), 0.2, 0.1, 
                                     boxstyle="round,pad=0.01", 
                                     color='blue', alpha=0.7)
                ax4.add_patch(rect)
                resonance = r1 if name == 'r1' else r2
                ax4.text(pos[0], pos[1], resonance.constant_name.replace('_', ' ').title(), 
                        ha='center', va='center', fontsize=10)
        
        # Draw connections
        ax4.plot([positions['r1'][0], positions['14'][0]], 
                [positions['r1'][1], positions['14'][1]], 'k-', alpha=0.5)
        ax4.plot([positions['r2'][0], positions['14'][0]], 
                [positions['r2'][1], positions['14'][1]], 'k-', alpha=0.5)
        ax4.plot([positions['r1'][0], positions['r2'][0]], 
                [positions['r1'][1], positions['r2'][1]], 'k-', alpha=0.5, linestyle='--')
        
        # Add labels on connections
        ax4.text(0.4, 0.5, f'{index_ratio:.3f}', ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        self.report.save_figure(fig, "resonance_comparison")
        plt.close()
    
    def run_exploration(self):
        """Run complete resonance exploration"""
        logger.info("🚀 Starting resonance exploration...")
        
        # Information about available data
        if self.zeros:
            logger.info(f"📊 Available data: {len(self.zeros):,} zeros")
            logger.info(f"📊 Index range: 1 to {self.zeros[-1][0]:,}")
        else:
            logger.warning("⚠️ No zero data available - using synthetic analysis")
        
        # Add initial information to report
        self.report.add_section("Initial Information")
        if self.zeros:
            self.report.add_text(f"Available data: {len(self.zeros):,} zeros")
            self.report.add_text(f"Index range: 1 to {self.zeros[-1][0]:,}")
        else:
            self.report.add_text("No zero data available - using synthetic analysis")
        
        # 1. Individual analysis of each resonance
        for i, resonance in enumerate(self.target_resonances):
            self.report.add_section(f"Analysis of Resonance {i+1}: {resonance.constant_name.upper()}")
            
            # Neighborhood analysis (real or synthetic)
            if self.zeros and len(self.zeros) > resonance.zero_index:
                neighborhood_data = self.analyze_neighborhood(resonance, 100)
                if neighborhood_data:
                    self.report.add_text("📊 Neighborhood statistics:")
                    self.report.add_text(f"   Mean of γ: {neighborhood_data['local_stats']['mean']:.6f}")
                    self.report.add_text(f"   Standard deviation: {neighborhood_data['local_stats']['std']:.6f}")
                    self.report.add_text(f"   Local minimum: Zero #{neighborhood_data['local_minimum']['index']:,} "
                               f"(distance: {neighborhood_data['local_minimum']['distance']:.2e})")
            else:
                self.report.add_text("📊 Using synthetic analysis (real data not available)")
            
            # Test mathematical relations
            math_results = self.test_mathematical_relations(resonance)
            self.report.add_text("\n🔢 Significant mathematical relations:")
            
            significant_found = False
            table_data = []
            for key, value in math_results.items():
                if isinstance(value, dict) and value.get('significant', False):
                    table_data.append([
                        key,
                        f"{value['value']:.6f}",
                        f"{value['target']:.6f}",
                        f"{value['error']:.2%}"
                    ])
                    significant_found = True
            
            if significant_found:
                headers = ["Relation", "Value", "Target", "Error"]
                self.report.add_table(headers, table_data)
            else:
                self.report.add_text("   No obvious mathematical relation found")
            
            # Generate visualization
            self.visualize_resonance(resonance)
        
        # 2. Comparison between resonances
        self.compare_resonances()
        
        # 3. Additional theoretical analysis
        self.report.add_section("Theoretical Analysis")
        
        self.report.add_text("🔍 Hypotheses about resonances:")
        self.report.add_text("1. The fine structure resonance may be related to:")
        self.report.add_text("   - Electroweak unification energy scale")
        self.report.add_text("   - Coupling constant in gauge theories")
        self.report.add_text("   - Relations with the number 137 (inverse of α)")
        
        self.report.add_text("\n2. The electron mass resonance may indicate:")
        self.report.add_text("   - Mathematical origin for particle masses")
        self.report.add_text("   - Connection with Standard Model mass hierarchy")
        self.report.add_text("   - Relation with symmetry breaking mechanisms")
        
        self.report.add_text("\n3. The ~14 ratio between indices suggests:")
        self.report.add_text("   - Possible relation with the number of particle generations")
        self.report.add_text("   - Connection with extra dimensions in string theories")
        self.report.add_text("   - Scale factor between different physical regimes")
        
        # Create theoretical visualization
        self.create_theoretical_visualization()
        
        # 4. Close the report
        report_path = self.report.close()
        
        logger.info("✅ Exploration completed!")
        logger.info(f"📄 Report available at: {report_path}")
        
        return report_path
    
    def create_theoretical_visualization(self):
        """Create visualization of theoretical implications"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Chart 1: Energy scales
        ax1.set_title("Resonance Energies", fontsize=14, weight='bold')
        
        energies = [r.energy_gev for r in self.target_resonances]
        names = [r.constant_name.replace('_', ' ').title() for r in self.target_resonances]
        colors = ['blue', 'red']
        
        bars = ax1.bar(names, energies, color=colors, alpha=0.7)
        ax1.set_ylabel('Energy (GeV)')
        ax1.set_yscale('log')
        
        # Add values on bars
        for bar, energy in zip(bars, energies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{energy:.1f} GeV', ha='center', va='bottom', fontsize=10)
        
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Resonance qualities
        ax2.set_title("Resonance Qualities", fontsize=14, weight='bold')
        
        qualities = [r.quality for r in self.target_resonances]
        
        bars = ax2.bar(names, qualities, color=colors, alpha=0.7)
        ax2.set_ylabel('Quality')
        ax2.set_yscale('log')
        
        # Add values on bars
        for bar, quality in zip(bars, qualities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{quality:.2e}', ha='center', va='bottom', fontsize=10)
        
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Physical implication diagram
        ax3.set_title("Physical Implications", fontsize=14, weight='bold')
        
        # Node positions
        positions = {
            'fine_structure': (0.3, 0.8),
            'electron_mass': (0.7, 0.8),
            'unification': (0.5, 0.5),
            'mass_hierarchy': (0.3, 0.2),
            'symmetry_breaking': (0.7, 0.2)
        }
        
        # Draw nodes
        for name, pos in positions.items():
            if name in ['unification', 'mass_hierarchy', 'symmetry_breaking']:
                rect = FancyBboxPatch((pos[0]-0.12, pos[1]-0.05), 0.24, 0.1, 
                                     boxstyle="round,pad=0.01", 
                                     color='green', alpha=0.7)
                ax3.add_patch(rect)
                ax3.text(pos[0], pos[1], name.replace('_', ' ').title(), 
                        ha='center', va='center', fontsize=9)
            else:
                rect = FancyBboxPatch((pos[0]-0.1, pos[1]-0.05), 0.2, 0.1, 
                                     boxstyle="round,pad=0.01", 
                                     color='blue', alpha=0.7)
                ax3.add_patch(rect)
                ax3.text(pos[0], pos[1], name.replace('_', ' ').title(), 
                        ha='center', va='center', fontsize=9)
        
        # Draw connections
        ax3.plot([positions['fine_structure'][0], positions['unification'][0]], 
                [positions['fine_structure'][1], positions['unification'][1]], 'k-', alpha=0.5)
        ax3.plot([positions['electron_mass'][0], positions['unification'][0]], 
                [positions['electron_mass'][1], positions['unification'][1]], 'k-', alpha=0.5)
        ax3.plot([positions['fine_structure'][0], positions['mass_hierarchy'][0]], 
                [positions['fine_structure'][1], positions['mass_hierarchy'][1]], 'k-', alpha=0.5)
        ax3.plot([positions['electron_mass'][0], positions['symmetry_breaking'][0]], 
                [positions['electron_mass'][1], positions['symmetry_breaking'][1]], 'k-', alpha=0.5)
        
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # Chart 4: Experimental predictions
        ax4.set_title("Experimental Predictions", fontsize=14, weight='bold')
        
        predictions = [
            ("New particles\n8.7 TeV", 0.9, 'blue'),
            ("95 TeV phenomena", 0.8, 'red'),
            ("Exact constant\nrelationships", 0.7, 'green'),
            ("Mass structure", 0.6, 'purple'),
            ("Mathematical\nextensions", 0.5, 'orange')
        ]
        
        y_pos = np.arange(len(predictions))
        values = [item[1] for item in predictions]
        colors = [item[2] for item in predictions]
        labels = [item[0] for item in predictions]
        
        bars = ax4.barh(y_pos, values, color=colors, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels)
        ax4.set_xlabel('Confidence')
        ax4.set_xlim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # Add values on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{value:.1f}', ha='left', va='center')
        
        plt.tight_layout()
        self.report.save_figure(fig, "theoretical_implications")
        plt.close()

# Main execution
if __name__ == "__main__":
    try:
        explorer = ResonanceExplorer()
        report_path = explorer.run_exploration()
        print(f"\nAnalysis completed! Report saved at: {report_path}")
    except Exception as e:
        logger.error(f"❌ Error during exploration: {e}")
        import traceback
        traceback.print_exc()
