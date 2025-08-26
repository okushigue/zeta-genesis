#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_dark_matter_results.py - Analysis of dark matter hunting results
Author: Jefferson M. Okushigue
Date: 2025-08-25
Analyzes and visualizes the results found by the Dark Matter Hunter
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
sns.set_palette("plasma")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

class DarkMatterAnalyzer:
    def __init__(self, results_dir="zvt_dark_matter_results"):
        self.results_dir = results_dir
        self.state_file = "session_state_dark_matter.json"
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
            
        print("\n" + "🌑" * 80)
        print("🏆 BEST DARK MATTER RESONANCES FOUND 🏆")
        print("🌑" * 80)
        
        # Dark matter constants for reference
        DARK_MATTER_CONSTANTS = {
            'dark_matter_density': 1.18e-26,          # Dark matter density (kg/m³)
            'omega_dm': 0.2589,                        # Dark matter density parameter
            'wimp_mass_light': 1e-3,                   # Light WIMP mass (GeV/c²)
            'wimp_mass_heavy': 1e3,                    # Heavy WIMP mass (GeV/c²)
            'axion_mass': 1e-5,                        # Axion mass (eV/c²)
            'sterile_neutrino_mass': 1e-3,             # Sterile neutrino mass (eV/c²)
            'primordial_black_hole_mass': 1e-12,       # Primordial black hole mass (M_sol)
            'self_interaction_cross_section': 1e-24,   # Self-interaction cross section (cm²/g)
            'dm_annihilation_rate': 3e-26,             # DM annihilation rate (cm³/s)
            'dm_scattering_cross_section': 1e-45,      # DM scattering cross section (cm²)
            'neutralino_mass': 100,                    # Neutralino mass (GeV/c²)
            'kaluza_klein_mass': 500,                  # Kaluza-Klein particle mass (GeV/c²)
            'gravitino_mass': 1e-3,                    # Gravitino mass (GeV/c²)
            'dark_photon_mass': 1e-3,                  # Dark photon mass (GeV/c²)
            'freeze_out_temperature': 20,              # Freeze-out temperature (GeV)
            'dm_velocity_dispersion': 220,             # DM velocity dispersion (km/s)
            'local_dm_density': 0.3,                   # Local DM density (GeV/cm³)
            'galactic_escape_velocity': 544,           # Galactic escape velocity (km/s)
            'mond_acceleration': 1.2e-10,              # MOND acceleration (m/s²)
            'bullet_cluster_constraint': 1e-24        # Bullet Cluster constraint (cm²/g)
        }
        
        descriptions = {
            'dark_matter_density': 'Average dark matter density in the universe',
            'omega_dm': 'Dark matter density parameter (Ω_dm)',
            'wimp_mass_light': 'Light WIMP (Weakly Interacting Massive Particle) mass',
            'wimp_mass_heavy': 'Heavy WIMP mass',
            'axion_mass': 'Theoretical axion mass',
            'sterile_neutrino_mass': 'Sterile neutrino mass',
            'primordial_black_hole_mass': 'Primordial black hole mass',
            'self_interaction_cross_section': 'Self-interaction cross section',
            'dm_annihilation_rate': 'Dark matter annihilation rate',
            'dm_scattering_cross_section': 'Scattering cross section',
            'neutralino_mass': 'Neutralino mass (supersymmetric candidate)',
            'kaluza_klein_mass': 'Kaluza-Klein particle mass',
            'gravitino_mass': 'Gravitino mass',
            'dark_photon_mass': 'Dark photon mass',
            'freeze_out_temperature': 'Thermal decoupling temperature',
            'dm_velocity_dispersion': 'Dark matter velocity dispersion',
            'local_dm_density': 'Local dark matter density',
            'galactic_escape_velocity': 'Milky Way escape velocity',
            'mond_acceleration': 'MOND acceleration scale',
            'bullet_cluster_constraint': 'Bullet Cluster constraint'
        }
        
        # Formatted table
        print("| Constant               | Zero #        | Gamma (γ)            | Quality      | Mass (GeV)   | Rel. Error (%) |")
        print("|-------------------------|---------------|----------------------|----------------|---------------|---------------|")
        
        for const_name, res in self.best_resonances.items():
            zero_num = res['zero_index']
            gamma = res['gamma']
            quality = res['quality']
            mass = gamma / 100  # Conversion to GeV (scale adjusted for dark matter)
            
            # Calculate relative error
            const_value = DARK_MATTER_CONSTANTS.get(const_name, 1.0)
            rel_error = (quality / const_value) * 100
            
            print(f"| {const_name[:23]:23s} | {zero_num:13,} | {gamma:20.15f} | {quality:.6e} | {mass:13.6f} | {rel_error:13.6e} |")
        
        print("\n📋 RESONANCE DETAILS:")
        print("=" * 80)
        
        for const_name, res in self.best_resonances.items():
            desc = descriptions.get(const_name, 'Unknown dark matter constant')
            const_value = DARK_MATTER_CONSTANTS.get(const_name, 1.0)
            rel_error = (res['quality'] / const_value) * 100
            
            print(f"\n🔬 {const_name.upper().replace('_', ' ')}:")
            print(f"   📖 Description: {desc}")
            print(f"   🎯 Zero #{res['zero_index']:,} (γ = {res['gamma']:.15f})")
            print(f"   💎 Quality: {res['quality']:.15e}")
            print(f"   📊 Relative error: {rel_error:.12e}%")
            print(f"   🔬 Tolerance: {res['tolerance']:.0e}")
            print(f"   ⚖️  Estimated mass: {res['gamma']/100:.6f} GeV/c²")
    
    def create_mass_spectrum_plot(self):
        """Creates a plot of the mass spectrum of resonances"""
        if not self.best_resonances:
            print("❌ No resonances to plot!")
            return
            
        # Prepare data
        constants = list(self.best_resonances.keys())
        masses = [res['gamma']/100 for res in self.best_resonances.values()]
        qualities = [res['quality'] for res in self.best_resonances.values()]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Mass Spectrum
        colors = plt.cm.plasma(np.linspace(0, 1, len(constants)))
        bars1 = ax1.bar(range(len(constants)), masses, color=colors, alpha=0.8)
        
        ax1.set_xlabel('Dark Matter Constants')
        ax1.set_ylabel('Estimated Mass (GeV/c²)')
        ax1.set_title('Mass Spectrum of Best Resonances\nZeta Function Zeros vs Dark Matter')
        ax1.set_xticks(range(len(constants)))
        ax1.set_xticklabels([c.replace('_', '\n').title() for c in constants], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better mass visualization
        
        # Add values on bars
        for i, (bar, mass) in enumerate(zip(bars1, masses)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{mass:.2e}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Plot 2: Resonance Quality (log scale)
        bars2 = ax2.bar(range(len(constants)), qualities, color=colors, alpha=0.8)
        ax2.set_yscale('log')
        ax2.set_xlabel('Dark Matter Constants')
        ax2.set_ylabel('Resonance Quality (log scale)')
        ax2.set_title('Quality of Best Resonances\n(Lower = Better)')
        ax2.set_xticks(range(len(constants)))
        ax2.set_xticklabels([c.replace('_', '\n').title() for c in constants], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.results_dir, f"mass_spectrum_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {plot_file}")
        
        plt.show()
    
    def create_particle_physics_significance_plot(self):
        """Creates a plot of significance in particle physics"""
        if not self.best_resonances:
            return
            
        # Constants grouped by category
        wimp_candidates = ['wimp_mass_light', 'wimp_mass_heavy', 'neutralino_mass']
        light_candidates = ['axion_mass', 'sterile_neutrino_mass', 'dark_photon_mass', 'gravitino_mass']
        exotic_candidates = ['kaluza_klein_mass', 'primordial_black_hole_mass']
        interactions = ['self_interaction_cross_section', 'dm_annihilation_rate', 
                       'dm_scattering_cross_section', 'bullet_cluster_constraint']
        cosmological = ['dark_matter_density', 'omega_dm', 'freeze_out_temperature', 
                       'local_dm_density', 'dm_velocity_dispersion', 'galactic_escape_velocity', 'mond_acceleration']
        
        categories = {
            'WIMP Candidates': wimp_candidates,
            'Light Particles': light_candidates,
            'Exotic Candidates': exotic_candidates,
            'Interactions': interactions,
            'Cosmological Properties': cosmological
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        fig.suptitle('Particle Physics Significance Analysis - Dark Matter', fontsize=16)
        
        # Prepare data by category
        for idx, (category, const_list) in enumerate(categories.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            cat_constants = [c for c in const_list if c in self.best_resonances]
            if not cat_constants:
                ax.text(0.5, 0.5, f'No resonances\nin {category}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(category)
                continue
                
            cat_masses = [self.best_resonances[c]['gamma']/100 for c in cat_constants]
            cat_qualities = [self.best_resonances[c]['quality'] for c in cat_constants]
            
            # Scatter plot mass vs quality
            colors = plt.cm.tab10(np.linspace(0, 1, len(cat_constants)))
            scatter = ax.scatter(cat_masses, cat_qualities, c=colors, s=100, alpha=0.7)
            
            # Add labels
            for i, const in enumerate(cat_constants):
                ax.annotate(const.replace('_', ' ').title(), 
                           (cat_masses[i], cat_qualities[i]),
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
            
            ax.set_xlabel('Estimated Mass (GeV/c²)')
            ax.set_ylabel('Quality (log scale)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f'{category}\nMass vs Quality')
            ax.grid(True, alpha=0.3)
        
        # Remove extra axes
        for idx in range(len(categories), len(axes)):
            axes[idx].remove()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.results_dir, f"particle_physics_significance_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Significance plot saved: {plot_file}")
        
        plt.show()
    
    def generate_summary_report(self):
        """Generates a summary report of the results"""
        if not self.best_resonances:
            print("❌ No resonances to report!")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.results_dir, f"SUMMARY_Dark_Matter_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("🌑 EXECUTIVE SUMMARY - DARK MATTER ZVT HUNTER 🌑\n")
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
                f.write(f"   Mass: {res['gamma']/100:.6f} GeV/c²\n\n")
            
            f.write("🌑 IMPLICATIONS FOR PARTICLE PHYSICS:\n")
            f.write("-" * 50 + "\n")
            f.write("• Possible connection between Riemann zeros and particle masses\n")
            f.write("• Critical mass scales for dark matter candidates\n")
            f.write("• Deep mathematical patterns in physics beyond the Standard Model\n")
            f.write("• Connection between number theory and cosmology\n")
            f.write("• New approaches for dark matter detection\n\n")
            
            f.write("🔬 SUGGESTED NEXT STEPS:\n")
            f.write("-" * 50 + "\n")
            f.write("1. Comparison with direct detection experiment results\n")
            f.write("2. Analysis of correlations with N-body simulation data\n")
            f.write("3. Extension to other theoretical candidates\n")
            f.write("4. Investigation of implications for supersymmetry\n")
            f.write("5. Collaboration with experimental groups (LHC, XENON, etc.)\n")
        
        print(f"📋 Executive report saved: {report_file}")
        return report_file
    
    def run_complete_analysis(self):
        """Runs complete analysis of the results"""
        print("🌑" * 60)
        print("🔍 DARK MATTER RESULTS ANALYZER")
        print("🌑" * 60)
        
        # Load data
        session_data = self.load_session_results()
        if not session_data:
            print("❌ Could not load session data!")
            return
        
        # Show results
        self.show_best_resonances()
        
        # Create visualizations
        print("\n📊 Generating visualizations...")
        self.create_mass_spectrum_plot()
        self.create_particle_physics_significance_plot()
        
        # Generate report
        print("\n📋 Generating executive report...")
        self.generate_summary_report()
        
        print("\n" + "🌑" * 60)
        print("✅ COMPLETE ANALYSIS OF RESULTS FINISHED!")
        print("🌑" * 60)

if __name__ == "__main__":
    analyzer = DarkMatterAnalyzer()
    analyzer.run_complete_analysis()
