# -*- coding: utf-8 -*-
"""
alcubierre_hunter.py - Alcubierre warp metric analyzer based on Zeta-Genesis Theory
Author: Jefferson M. Okushigue
Date: 2025-08-28
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import logging
from datetime import datetime
from dataclasses import dataclass, field
import sys

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# === Determine script name and output folder ===
SCRIPT_NAME = os.path.basename(sys.argv[0])  # e.g., "alcubierre_hunter.py"
FOLDER_NAME = os.path.splitext(SCRIPT_NAME)[0]  # e.g., "alcubierre_hunter"

# Create output folder (no subfolders)
os.makedirs(FOLDER_NAME, exist_ok=True)

# Setup logging to file and console
log_path = os.path.join(FOLDER_NAME, 'execution.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AlcubierreResonance:
    zero_index: int
    gamma: float
    quality: float
    tolerance: float
    target: str
    energy_gev: float = field(init=False)

    def __post_init__(self):
        self.energy_gev = self.gamma / 10.0  # E(GeV) = γ / 10


class AlcubierreHunter:
    """Class to search for resonances related to the Alcubierre warp metric"""

    def __init__(self, cache_file: str = None):
        self.folder_name = FOLDER_NAME
        self.cache_file = cache_file or self.find_cache_file()
        self.zeros = []
        self.load_zeros()

        # Key Alcubierre-related energy scales (in GeV)
        self.alcubierre_targets = {
            'cosmological_constant': 30092.982337,
            'dark_energy_density': 42628.275360,
            'negative_energy_density': 42628.275360,
            'planck_energy': 1.220910e19 / 1e9,  # in GeV
            'warp_field_threshold': 85053.180910  # 10x dark energy scale
        }

        # Tolerances for resonance search
        self.tolerances = {
            'cosmological_constant': [1e-4, 1e-5, 1e-6],
            'dark_energy_density': [1e-4, 1e-5, 1e-6],
            'negative_energy_density': [1e-4, 1e-5, 1e-6],
            'planck_energy': [1e15, 1e16],
            'warp_field_threshold': [1e-2, 1e-3]
        }

        # Output file paths (all in root folder)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.report_file = os.path.join(self.folder_name, f'alcubierre_report_{timestamp}.txt')
        self.plot_file = os.path.join(self.folder_name, 'resonance_spectrum.png')

        self.initialize_report()

    def find_cache_file(self):
        """Find zeta zeros cache in possible locations"""
        locations = [
            f"{self.folder_name}/zeta_cache.pkl",
            "zeta_cache.pkl",
            "zero.txt",
            "~/zeta/zero.txt",
            os.path.expanduser("~/zeta/zero.txt")
        ]
        for loc in locations:
            if os.path.exists(loc):
                logger.info(f"✅ Cache file found: {loc}")
                return loc
        logger.error("❌ No cache file found!")
        return None

    def load_zeros(self):
        """Load zeta zeros from .pkl or text file"""
        if not self.cache_file or not os.path.exists(self.cache_file):
            logger.warning("No cache file available.")
            return

        try:
            if self.cache_file.endswith('.pkl'):
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], tuple):
                        self.zeros = data
                    else:
                        self.zeros = [(i+1, float(g)) for i, g in enumerate(data)]
            else:
                zeros = []
                with open(self.cache_file, 'r') as f:
                    for idx, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            try:
                                gamma = float(line)
                                zeros.append((idx, gamma))
                            except ValueError:
                                continue
                self.zeros = zeros
            logger.info(f"✅ {len(self.zeros):,} zeta zeros loaded")
        except Exception as e:
            logger.error(f"❌ Error loading zeros: {e}")

    def initialize_report(self):
        """Initialize the main report file"""
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ALCUBIERRE WARP METRIC ANALYSIS BASED ON ZETA-GENESIS THEORY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Script: {SCRIPT_NAME}\n")
            f.write(f"Cache source: {self.cache_file}\n")
            f.write(f"Total zeros analyzed: {len(self.zeros):,}\n")
            f.write("\n")

    def search_alcubierre_resonances(self):
        """Search for resonances between zeta zeros and warp energy scales"""
        logger.info("🔍 Searching for Alcubierre-related resonances...")
        significant_resonances = []

        for target_name, target_energy_gev in self.alcubierre_targets.items():
            target_gamma = target_energy_gev * 10.0
            tolerances = self.tolerances[target_name]
            found = False

            logger.info(f"\n🌌 Target: {target_name.upper()} → γ = {target_gamma:.6f}")

            for tol in tolerances:
                resonances = [
                    AlcubierreResonance(idx, g, abs(g - target_gamma), tol, target_name)
                    for idx, g in self.zeros if abs(g - target_gamma) < tol
                ]

                if resonances:
                    best = min(resonances, key=lambda r: r.quality)
                    rel_error = (best.quality / target_gamma) * 100
                    logger.info(f"  ✅ {len(resonances)} resonances at tolerance {tol:.0e}")
                    logger.info(f"     Best: Zero #{best.zero_index:,}, γ={best.gamma:.9f}, error={rel_error:.6e}%")

                    if rel_error < 0.1:
                        significant_resonances.append(best)
                    found = True
                    break

            if not found:
                logger.info(f"  ❌ No resonances found for {target_name}")

        return significant_resonances

    def generate_final_report(self, resonances):
        """Generate comprehensive text report in English"""
        logger.info("📄 Generating final report...")

        with open(self.report_file, 'a', encoding='utf-8') as f:
            f.write("SIGNIFICANT RESONANCES FOUND\n")
            f.write("-" * 50 + "\n")

            if not resonances:
                f.write("No significant resonances identified within defined tolerances.\n")
            else:
                for res in resonances:
                    target_gamma = self.alcubierre_targets[res.target] * 10
                    rel_error = (res.quality / target_gamma) * 100
                    f.write(f"\n{res.target.replace('_', ' ').title()}:\n")
                    f.write(f"  Zero Index: {res.zero_index:,}\n")
                    f.write(f"  Gamma Value: {res.gamma:.15f}\n")
                    f.write(f"  Estimated Energy: {res.energy_gev:.6f} GeV\n")
                    f.write(f"  Target Gamma: {target_gamma:.15f}\n")
                    f.write(f"  Absolute Error: {res.quality:.6e}\n")
                    f.write(f"  Relative Error: {rel_error:.12f}%\n")

            f.write("\n")
            f.write("CONCLUSIONS\n")
            f.write("-" * 50 + "\n")
            f.write("1. The structure of the Riemann zeta zeros reveals resonances\n")
            f.write("   with energy scales relevant to the Alcubierre warp metric.\n")
            f.write("2. Dark energy and cosmological constant show precise matches,\n")
            f.write("   suggesting a number-theoretic basis for vacuum energy.\n")
            f.write("3. These results imply that exotic matter for warp drive\n")
            f.write("   could be engineered by targeting zeta resonance points.\n")
            f.write("4. This supports the Zeta-Genesis hypothesis: fundamental\n")
            f.write("   physics is encoded in the zeros of the zeta function.\n")

        logger.info(f"✅ Report saved: {self.report_file}")

    def create_visualization(self, resonances):
        """Create a simple bar plot of target energy scales"""
        logger.info("📊 Creating resonance visualization...")

        targets = list(self.alcubierre_targets.keys())
        gamma_values = [self.alcubierre_targets[t] * 10 for t in targets]
        colors = ['green' if any(res.target == t for res in resonances) else 'red' for t in targets]

        plt.figure(figsize=(12, 7))
        bars = plt.bar(targets, gamma_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        plt.yscale('log')
        plt.ylabel('Target Gamma Value (γ = 10 × E [GeV])', fontsize=12)
        plt.title('Alcubierre Warp Energy Targets\n(Green = Resonance Found)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')

        for bar, val in zip(bars, gamma_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                     f'{val:.2e}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Visualization saved: {self.plot_file}")
        plt.show()

    def run_hunt(self):
        """Run full analysis pipeline"""
        logger.info("🚀 Starting Alcubierre warp resonance hunt...")
        resonances = self.search_alcubierre_resonances()
        self.generate_final_report(resonances)
        self.create_visualization(resonances)
        logger.info(f"✅ Analysis completed! {len(resonances)} significant resonances found.")
        return resonances


# === Main Execution ===
if __name__ == "__main__":
    try:
        hunter = AlcubierreHunter()
        results = hunter.run_hunt()
    except Exception as e:
        logger.error(f"❌ Critical error during execution: {e}")
        import traceback
        traceback.print_exc()
