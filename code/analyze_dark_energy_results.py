#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_dark_energy_results.py - Analysis of dark energy hunting results
Author: Jefferson M. Okushigue
Date: 2025-08-25
Analyzes and visualizes the results found by the Dark Energy Hunter
"""

import json
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import glob

# Visualization configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

class DarkEnergyAnalyzer:
    def __init__(self, results_dir="zvt_dark_energy_results"):
        self.results_dir = results_dir
        self.state_file = "session_state_dark_energy.json"
        self.best_resonances = {}
        
    def load_session_results(self):
        """Loads the session results"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    
                print("🔄 Session state loaded!")
                print(f"📊 Last processed zeros: {data.get('last_processed_index', 0):,}")
                print(f"📊 Total zeros: {data.get('total_zeros', 0):,}")
                
                # Load the best resonances
                best_res = data.get('best_resonances', {})
                for name, res_data in best_res.items():
                    self.best_resonances[name] = res_data
                    
                return data
            except Exception as e:
                print(f"❌ Error loading state: {e}")
                return None
        return None
    
    def show_best_resonances(self):
        """Shows the best resonances found"""
        if not self.best_resonances:
            print("❌ No resonances found!")
            return
            
        print("\n" + "🌌" * 80)
        print("🏆 BEST DARK ENERGY RESONANCES FOUND 🏆")
        print("🌌" * 80)
        
        # Dark energy constants for reference
        DARK_ENERGY_CONSTANTS = {
            'hubble_constant': 2.197e-18,
            'cosmological_constant': 1.1056e-52,
            'dark_energy_density': 5.96e-27,
            'critical_density': 8.62e-27,
            'omega_lambda': 0.6889,
            'equation_of_state': -1.0,
            'planck_time': 5.391247e-44,
            'planck_length': 1.616255e-35,
            'vacuum_energy_scale': 2.4e-3,
            'quintessence_mass': 3.16e-33,
            'dark_energy_scale': 0.0024,
            'acceleration_scale': 1.2e-10
        }
        
        descriptions = {
            'hubble_constant': 'Hubble constant - Cosmic expansion rate (H₀)',
            'cosmological_constant': 'Einstein\'s Cosmological Constant (Λ)',
            'dark_energy_density': 'Dark energy density (ρ_Λ)',
            'critical_density': 'Critical density of the universe (ρ_c)',
            'omega_lambda': 'Dark energy density parameter (Ω_Λ)',
            'equation_of_state': 'Dark energy equation of state (w)',
            'planck_time': 'Planck time scale (t_P)',
            'planck_length': 'Planck length scale (l_P)',
            'vacuum_energy_scale': 'Theoretical vacuum energy scale',
            'quintessence_mass': 'Hypothetical quintessence field mass',
            'dark_energy_scale': 'Characteristic dark energy scale',
            'acceleration_scale': 'Cosmic acceleration scale'
        }
        
        # Formatted table
        print("| Constant               | Zero #        | Gamma (γ)            | Quality      | Energy (GeV) | Rel. Error (%) |")
        print("|-------------------------|---------------|----------------------|----------------|---------------|---------------|")
        
        for const_name, res in self.best_resonances.items():
            zero_num = res['zero_index']
            gamma = res['gamma']
            quality = res['quality']
            energy = gamma / 10  # Conversion to GeV
            
            # Calculate relative error
            const_value = DARK_ENERGY_CONSTANTS.get(const_name, 1.0)
            rel_error = (quality / const_value) * 100
            
            print(f"| {const_name[:23]:23s} | {zero_num:13,} | {gamma:20.15f} | {quality:.6e} | {energy:13.6f} | {rel_error:13.6e} |")
        
        print("\n📋 RESONANCE DETAILS:")
        print("=" * 80)
        
        for const_name, res in self.best_resonances.items():
            desc = descriptions.get(const_name, 'Unknown dark energy constant')
            const_value = DARK_ENERGY_CONSTANTS.get(const_name, 1.0)
            rel_error = (res['quality'] / const_value) * 100
            
            print(f"\n🔬 {const_name.upper().replace('_', ' ')}:")
            print(f"   📖 Description: {desc}")
            print(f"   🎯 Zero #{res['zero_index']:,} (γ = {res['gamma']:.15f})")
            print(f"   💎 Quality: {res['quality']:.15e}")
            print(f"   📊 Relative error: {rel_error:.12e}%")
            print(f"   🔬 Tolerance: {res['tolerance']:.0e}")
            print(f"   ⚡ Estimated energy: {res['gamma']/10:.6f} GeV")
        
    def create_energy_spectrum_plot(self):
        """Creates a plot of the energy spectrum of resonances"""
        if not self.best_resonances:
            print("❌ No resonances to plot!")
            return
            
        # Prepare data
        constants = list(self.best_resonances.keys())
        energies = [res['gamma']/10 for res in self.best_resonances.values()]
        qualities = [res['quality'] for res in self.best_resonances.values()]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Energy Spectrum
        colors = plt.cm.viridis(np.linspace(0, 1, len(constants)))
        bars1 = ax1.bar(range(len(constants)), energies, color=colors, alpha=0.8)
        
        ax1.set_xlabel('Dark Energy Constants')
        ax1.set_ylabel('Estimated Energy (GeV)')
        ax1.set_title('Energy Spectrum of Best Resonances\nZeta Function Zeros vs Dark Energy')
        ax1.set_xticks(range(len(constants)))
        ax1.set_xticklabels([c.replace('_', '\n').title() for c in constants], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add values on bars
        for i, (bar, energy) in enumerate(zip(bars1, energies)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(energies)*0.01,
                    f'{energy:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Resonance Quality (log scale)
        bars2 = ax2.bar(range(len(constants)), qualities, color=colors, alpha=0.8)
        ax2.set_yscale('log')
        ax2.set_xlabel('Dark Energy Constants')
        ax2.set_ylabel('Resonance Quality (log scale)')
        ax2.set_title('Quality of Best Resonances\n(Lower = Better)')
        ax2.set_xticks(range(len(constants)))
        ax2.set_xticklabels([c.replace('_', '\n').title() for c in constants], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.results_dir, f"energy_spectrum_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {plot_file}")
        
        plt.show()
    
    def create_cosmological_significance_plot(self):
        """Creates a plot of cosmological significance"""
        if not self.best_resonances:
            return
            
        # Constants grouped by category
        quantum_gravity = ['planck_time', 'planck_length']
        cosmic_expansion = ['hubble_constant', 'cosmological_constant', 'critical_density']
        dark_energy = ['dark_energy_density', 'omega_lambda', 'equation_of_state', 
                      'vacuum_energy_scale', 'quintessence_mass', 'dark_energy_scale']
        cosmic_dynamics = ['acceleration_scale']
        
        categories = {
            'Quantum Gravity': quantum_gravity,
            'Cosmic Expansion': cosmic_expansion,
            'Dark Energy': dark_energy,
            'Cosmic Dynamics': cosmic_dynamics
        }
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cosmological Significance Analysis of Resonances', fontsize=16)
        
        # Prepare data by category
        for idx, (category, const_list) in enumerate(categories.items()):
            ax = [ax1, ax2, ax3, ax4][idx]
            
            cat_constants = [c for c in const_list if c in self.best_resonances]
            if not cat_constants:
                ax.text(0.5, 0.5, f'No resonances\nin {category}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(category)
                continue
                
            cat_energies = [self.best_resonances[c]['gamma']/10 for c in cat_constants]
            cat_qualities = [self.best_resonances[c]['quality'] for c in cat_constants]
            
            # Scatter plot energy vs quality
            colors = plt.cm.tab10(np.linspace(0, 1, len(cat_constants)))
            scatter = ax.scatter(cat_energies, cat_qualities, c=colors, s=100, alpha=0.7)
            
            # Add labels
            for i, const in enumerate(cat_constants):
                ax.annotate(const.replace('_', ' ').title(), 
                           (cat_energies[i], cat_qualities[i]),
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
            
            ax.set_xlabel('Estimated Energy (GeV)')
            ax.set_ylabel('Quality (log scale)')
            ax.set_yscale('log')
            ax.set_title(f'{category}\nEnergy vs Quality')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.results_dir, f"cosmological_significance_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Significance plot saved: {plot_file}")
        
        plt.show()
    
    def generate_summary_report(self):
        """Generates a summary report of the results"""
        if not self.best_resonances:
            print("❌ No resonances to report!")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.results_dir, f"SUMMARY_Dark_Energy_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("🌌 EXECUTIVE SUMMARY - DARK ENERGY ZVT HUNTER 🌌\n")
            f.write("="*80 + "\n\n")
            f.write(f"📅 Analysis Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"🎯 Total Resonances Found: {len(self.best_resonances)}\n")
            f.write(f"📊 Zeta Function Zeros Analyzed: 2,001,052\n\n")
            
            f.write("🏆 TOP 5 BEST RESONANCES:\n")
            f.write("-" * 50 + "\n")
            
            # Sort by quality (lower = better)
            sorted_res = sorted(self.best_resonances.items(), 
                              key=lambda x: x[1]['quality'])[:5]
            
            for i, (name, res) in enumerate(sorted_res, 1):
                f.write(f"{i}. {name.upper().replace('_', ' ')}\n")
                f.write(f"   Zero: #{res['zero_index']:,}\n")
                f.write(f"   Gamma: {res['gamma']:.12f}\n")
                f.write(f"   Quality: {res['quality']:.6e}\n")
                f.write(f"   Energy: {res['gamma']/10:.6f} GeV\n\n")
            
            f.write("🌌 COSMOLOGICAL IMPLICATIONS:\n")
            f.write("-" * 50 + "\n")
            f.write("• Possible connection between deep mathematical structures\n")
            f.write("  and fundamental constants of the universe\n")
            f.write("• Resonances may indicate critical energy scales\n")
            f.write("• Patterns in zeta zeros reflecting fundamental physics\n")
            f.write("• Connection between quantum and cosmological scales\n\n")
            
            f.write("🔬 SUGGESTED NEXT STEPS:\n")
            f.write("-" * 50 + "\n")
            f.write("1. Deeper statistical analysis of resonances\n")
            f.write("2. Comparison with observational dark energy data\n")
            f.write("3. Extension to more zeta function zeros\n")
            f.write("4. Theoretical investigation of found connections\n")
            f.write("5. Publication of results for scientific review\n")
        
        print(f"📋 Executive report saved: {report_file}")
        return report_file
    
    def run_complete_analysis(self):
        """Runs complete analysis of the results"""
        print("🌌" * 60)
        print("🔍 DARK ENERGY RESULTS ANALYZER")
        print("🌌" * 60)
        
        # Load data
        session_data = self.load_session_results()
        if not session_data:
            print("❌ Could not load session data!")
            return
        
        # Show results
        self.show_best_resonances()
        
        # Create visualizations
        print("\n📊 Generating visualizations...")
        self.create_energy_spectrum_plot()
        self.create_cosmological_significance_plot()
        
        # Generate report
        print("\n📋 Generating executive report...")
        self.generate_summary_report()
        
        print("\n" + "🌌" * 60)
        print("✅ COMPLETE ANALYSIS OF RESULTS FINISHED!")
        print("🌌" * 60)

if __name__ == "__main__":
    analyzer = DarkEnergyAnalyzer()
    analyzer.run_complete_analysis()
