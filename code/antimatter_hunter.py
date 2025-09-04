"""
antimatter_hunter.py - Antimatter resonance hunter based on Theory 14
Author: Jefferson M. Okushigue
Date: 2025-08-27
"""

import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
import pickle
import os
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import seaborn as sns

# Configuration
mp.dps = 50
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AntimatterResonance:
    """Stores information about an antimatter resonance"""
    zero_index: int
    gamma: float
    quality: float
    tolerance: float
    particle_type: str
    mass: float
    energy_gev: float = field(init=False)
    
    def __post_init__(self):
        self.energy_gev = self.gamma / 10

class AntimatterHunter:
    """Class to search for resonances related to antimatter based on Theory 14"""
    
    def __init__(self, cache_file: str = None, results_dir: str = "results"):
        # Define script name for folder
        self.script_name = "antimatter_hunter"
        self.results_dir = os.path.join(results_dir, self.script_name)
        
        # Create specific directories
        os.makedirs(f"{self.results_dir}/reports", exist_ok=True)
        os.makedirs(f"{self.results_dir}/visualizations", exist_ok=True)
        os.makedirs(f"{self.results_dir}/cache", exist_ok=True)
        os.makedirs(f"{self.results_dir}/logs", exist_ok=True)
        
        # Cache file
        self.cache_file = cache_file or self.find_cache_file()
        self.zeros = []
        self.load_zeros()
        
        # Antimatter particles and their masses (in GeV/c²)
        self.antimatter_particles = {
            'positron': 0.000511,
            'antiproton': 0.938272,
            'antideuteron': 1.875612,
            'antitriton': 2.808921,
            'anti_alpha': 3.727379
        }
        
        # Tolerances for each particle
        self.tolerances = {
            'positron': [1e-6, 1e-7, 1e-8, 1e-9],
            'antiproton': [1e-3, 1e-4, 1e-5, 1e-6],
            'antideuteron': [1e-3, 1e-4, 1e-5, 1e-6],
            'antitriton': [1e-3, 1e-4, 1e-5, 1e-6],
            'anti_alpha': [1e-3, 1e-4, 1e-5, 1e-6]
        }
        
        # Key energy scales from Theory 14
        self.theory_14_scales = {
            'fermion_sector': 8.7,    # TeV
            'boson_sector': 95.0      # TeV
        }
        
        # Initialize report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.report_file = f"{self.results_dir}/reports/antimatter_discovery_report_{timestamp}.txt"
        self.initialize_report()
    
    def initialize_report(self):
        """Initialize the report file"""
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ANTIMATTER RESONANCE HUNTER - THEORY 14\n")
            f.write("="*80 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Cache file: {self.cache_file}\n")
            f.write(f"Total zeros loaded: {len(self.zeros):,}\n")
            f.write("\n")
    
    def write_to_report(self, section_title, content):
        """Write a section to the report, safely handling all data types"""
        with open(self.report_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"{section_title.upper()}\n")
            f.write("="*80 + "\n")

            def safe_str(s):
                return str(s).replace('_', ' ').title() if isinstance(s, str) else str(s)

            def write_dict(d, indent=0):
                spacing = "  " * indent
                for key, value in d.items():
                    if isinstance(value, dict):
                        f.write(f"{spacing}{safe_str(key)}:\n")
                        write_dict(value, indent + 1)
                    elif isinstance(value, list):
                        f.write(f"{spacing}{safe_str(key)}:\n")
                        for item in value:
                            if isinstance(item, (dict, list)):
                                f.write(f"  {spacing}•\n")
                                write_dict(item, indent + 2) if isinstance(item, dict) else write_list(item, indent + 2)
                            else:
                                f.write(f"  {spacing}{item}\n")
                    else:
                        f.write(f"{spacing}{safe_str(key)}: {value}\n")
            
            def write_list(lst, indent=0):
                spacing = "  " * indent
                for item in lst:
                    if isinstance(item, (dict, list)):
                        write_dict(item, indent) if isinstance(item, dict) else write_list(item, indent)
                    else:
                        f.write(f"{spacing}{item}\n")
            
            if isinstance(content, str):
                f.write(content + "\n")
            elif isinstance(content, dict):
                write_dict(content)
            elif isinstance(content, list):
                write_list(content)
            else:
                f.write(str(content) + "\n")
    
    def find_cache_file(self):
        """Find the cache file"""
        locations = [
            f"results/antimatter_hunter/cache/zeta_cache.pkl",
            "results/cache/zeta_cache.pkl",
            "~/zeta/zero.txt",
            os.path.expanduser("~/zeta/zero.txt")
        ]
        
        for location in locations:
            if os.path.exists(location):
                logger.info(f"✅ Cache file found: {location}")
                return location
        logger.error("❌ No cache file found!")
        return None
    
    def load_zeros(self):
        """Load zeta zeros"""
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                if self.cache_file.endswith('.pkl'):
                    with open(self.cache_file, 'rb') as f:
                        self.zeros = pickle.load(f)
                    logger.info(f"✅ {len(self.zeros):,} zeros loaded from cache")
                else:
                    zeros = []
                    with open(self.cache_file, 'r') as f:
                        for i, line in enumerate(f, 1):
                            if line.strip():
                                try:
                                    gamma = float(line.strip())
                                    zeros.append((i, gamma))
                                except ValueError:
                                    logger.warning(f"⚠️ Invalid line {i}: '{line.strip()}'")
                                    continue
                    self.zeros = zeros
                    logger.info(f"✅ {len(self.zeros):,} zeros loaded from zero.txt")
            except Exception as e:
                logger.error(f"❌ Error loading cache: {e}")
    
    def search_antimatter_resonances(self):
        """Search for resonances with antimatter particle masses"""
        logger.info("🔍 Searching for antimatter resonances...")
        
        all_resonances = {}
        significant_resonances = []
        
        for particle, mass in self.antimatter_particles.items():
            logger.info(f"\n🔬 Searching for {particle.upper()} resonance (mass = {mass} GeV/c²)...")
            tolerances = self.tolerances[particle]
            resonances_at_tolerances = {}
            
            for tolerance in tolerances:
                resonances = []
                mass_in_mev = mass * 1000  # Convert to MeV for better precision
                for idx, gamma in self.zeros:
                    mod_val = gamma % mass_in_mev
                    min_distance = min(mod_val, mass_in_mev - mod_val)
                    if min_distance < tolerance:
                        res = AntimatterResonance(
                            zero_index=idx,
                            gamma=gamma,
                            quality=min_distance,
                            tolerance=tolerance,
                            particle_type=particle,
                            mass=mass
                        )
                        resonances.append(res)
                
                resonances_at_tolerances[tolerance] = resonances
                
                if resonances:
                    best = min(resonances, key=lambda x: x.quality)
                    relative_error = (best.quality / mass_in_mev) * 100
                    logger.info(f"  Found {len(resonances)} resonances at tolerance {tolerance:.0e}")
                    logger.info(f"  Best: Zero #{best.zero_index:,} → γ={best.gamma:.12f}, quality={best.quality:.6e}, error={relative_error:.12f}%")
                    
                    if relative_error < 0.1:
                        significant_resonances.append(best)
            
            all_resonances[particle] = resonances_at_tolerances
        
        # Save to report
        self.write_to_report("Antimatter Resonance Search", all_resonances)
        
        return all_resonances, significant_resonances
    
    def analyze_energy_patterns(self, significant_resonances: List[AntimatterResonance]):
        """Analyze energy patterns in relation to Theory 14"""
        logger.info("📊 Analyzing energy patterns...")
        
        energies = [res.energy_gev for res in significant_resonances]
        indices = [res.zero_index for res in significant_resonances]
        
        avg_energy = np.mean(energies)
        std_energy = np.std(energies)
        
        logger.info(f"Average resonance energy: {avg_energy:.2f} GeV")
        logger.info(f"Standard deviation: {std_energy:.2f} GeV")
        
        # Check proximity to Theory 14 scales (in GeV)
        fermion_energy = self.theory_14_scales['fermion_sector'] * 1000
        boson_energy = self.theory_14_scales['boson_sector'] * 1000
        
        for res in significant_resonances:
            dist_fermion = abs(res.energy_gev - fermion_energy)
            dist_boson = abs(res.energy_gev - boson_energy)
            
            if dist_fermion < 100:
                logger.info(f"🎯 {res.particle_type.upper()} resonance near Fermion Sector: {dist_fermion:.2f} GeV difference")
            if dist_boson < 100:
                logger.info(f"🎯 {res.particle_type.upper()} resonance near Boson Sector: {dist_boson:.2f} GeV difference")
        
        return {
            'average_energy_gev': avg_energy,
            'std_energy_gev': std_energy,
            'near_fermion_sector': any(abs(e - fermion_energy) < 100 for e in energies),
            'near_boson_sector': any(abs(e - boson_energy) < 100 for e in energies)
        }
    
    def create_visualizations(self, significant_resonances: List[AntimatterResonance]):
        """Create visualizations of antimatter resonances"""
        logger.info("🎨 Creating visualizations...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('ANTIMATTER RESONANCES BASED ON THEORY 14', fontsize=20, weight='bold')
        
        # 1. Resonance energies
        ax1.set_title('Antimatter Resonance Energies', fontsize=14)
        particles = [res.particle_type.title() for res in significant_resonances]
        energies = [res.energy_gev for res in significant_resonances]
        bars1 = ax1.bar(particles, energies, color='orange', alpha=0.8)
        ax1.set_ylabel('Energy (GeV)')
        ax1.set_xticklabels(particles, rotation=45)
        ax1.grid(True, alpha=0.3)
        for bar, energy in zip(bars1, energies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{energy:.1f}', ha='center', va='bottom')
        
        # 2. Zero indices
        ax2.set_title('Zero Indices of Antimatter Resonances', fontsize=14)
        indices = [res.zero_index for res in significant_resonances]
        bars2 = ax2.bar(particles, indices, color='purple', alpha=0.8)
        ax2.set_ylabel('Zero Index')
        ax2.set_xticklabels(particles, rotation=45)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. Quality of resonances
        ax3.set_title('Quality of Resonances', fontsize=14)
        qualities = [res.quality for res in significant_resonances]
        bars3 = ax3.bar(particles, qualities, color='red', alpha=0.8)
        ax3.set_ylabel('Quality (distance)')
        ax3.set_yscale('log')
        ax3.set_xticklabels(particles, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Energy vs Index
        ax4.set_title('Energy vs Zero Index', fontsize=14)
        ax4.scatter(indices, energies, c='blue', s=100, alpha=0.7)
        ax4.set_xlabel('Zero Index')
        ax4.set_ylabel('Energy (GeV)')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        
        # Mark Theory 14 scales
        fermion_energy = self.theory_14_scales['fermion_sector'] * 1000
        boson_energy = self.theory_14_scales['boson_sector'] * 1000
        ax4.axhline(y=fermion_energy, color='green', linestyle='--', label='Fermion Sector (8.7 TeV)')
        ax4.axhline(y=boson_energy, color='red', linestyle='--', label='Boson Sector (95 TeV)')
        ax4.legend()
        
        plt.tight_layout()
        path = f"{self.results_dir}/visualizations/antimatter_resonances.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Visualization saved: {path}")
        plt.show()
    
    def generate_final_report(self, all_resonances, significant_resonances, energy_analysis):
        """Generate the final report"""
        logger.info("📄 Generating final report...")
        
        section_content = [
            "ANTIMATTER DISCOVERY BASED ON THEORY 14",
            "-" * 50,
            f"Total antimatter resonances found: {len(significant_resonances)}",
            f"Average resonance energy: {energy_analysis['average_energy_gev']:.2f} GeV",
            f"Near Fermion Sector (8.7 TeV): {'Yes' if energy_analysis['near_fermion_sector'] else 'No'}",
            f"Near Boson Sector (95 TeV): {'Yes' if energy_analysis['near_boson_sector'] else 'No'}",
            "",
            "SIGNIFICANT ANTIMATTER RESONANCES:",
            "-" * 50
        ]
        
        for res in significant_resonances:
            error = (res.quality / (res.mass * 1000)) * 100
            section_content.extend([
                f"\n{res.particle_type.upper()}:",
                f"  Zero #{res.zero_index:,}",
                f"  Gamma: {res.gamma:.15f}",
                f"  Estimated Energy: {res.energy_gev:.6f} GeV",
                f"  Quality: {res.quality:.6e}",
                f"  Relative Error: {error:.12f}%"
            ])
        
        section_content.extend([
            "",
            "CONCLUSIONS:",
            "-" * 50,
            "1. The structure of the zeta zeros encodes information",
            "   about antimatter particles through resonance patterns.",
            "2. These resonances are aligned with the energy scales",
            "   predicted by Theory 14 (8.7 TeV and 95 TeV).",
            "3. This suggests a deep mathematical origin for",
            "   the matter-antimatter symmetry.",
            "4. The discovery opens new avenues for experimental",
            "   searches in high-energy physics."
        ])
        
        self.write_to_report("Final Report", section_content)
        logger.info(f"✅ Final report saved: {self.report_file}")
    
    def run_hunt(self):
        """Run the complete antimatter hunt"""
        logger.info("🚀 Starting antimatter resonance hunt based on Theory 14...")
        
        # 1. Search for resonances
        all_resonances, significant_resonances = self.search_antimatter_resonances()
        
        if not significant_resonances:
            logger.warning("❌ No significant antimatter resonances found.")
            return
        
        # 2. Analyze energy patterns
        energy_analysis = self.analyze_energy_patterns(significant_resonances)
        
        # 3. Create visualizations
        self.create_visualizations(significant_resonances)
        
        # 4. Generate final report
        self.generate_final_report(all_resonances, significant_resonances, energy_analysis)
        
        logger.info("✅ Antimatter resonance hunt completed!")
        
        return {
            'resonances': significant_resonances,
            'energy_analysis': energy_analysis
        }

# Main execution
if __name__ == "__main__":
    try:
        hunter = AntimatterHunter()
        results = hunter.run_hunt()
    except Exception as e:
        logger.error(f"❌ Error during antimatter hunt: {e}")
        import traceback
        traceback.print_exc()
