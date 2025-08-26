#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dark_energy_hunter.py - Zeta/Dark Energy resonance hunter
Author: Jefferson M. Okushigue
Date: 2025-08-25
Searches for resonances with dark energy related constants
"""
import numpy as np
from mpmath import mp
import time
import concurrent.futures
import pickle
import os
import signal
import sys
from datetime import datetime
from scipy import stats
import warnings
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import psutil
from tqdm import tqdm
import json

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("zvt_dark_energy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
mp.dps = 50  # High precision

# Visualization configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

@dataclass
class Resonance:
    """Class to store information about a resonance"""
    zero_index: int
    gamma: float
    quality: float
    tolerance: float
    constant_name: str
    constant_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'zero_index': self.zero_index,
            'gamma': self.gamma,
            'quality': self.quality,
            'tolerance': self.tolerance,
            'constant_name': self.constant_name,
            'constant_value': self.constant_value,
            'energy_gev': self.gamma / 10
        }

@dataclass
class SessionState:
    """Class to store session state for resumption"""
    last_processed_index: int = 0
    best_resonances: Dict[str, Resonance] = field(default_factory=dict)
    session_results: List[Any] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    total_zeros: int = 0

class ZetaDarkEnergyHunter:
    """Main class for analyzing resonances between zeta zeros and dark energy constants"""
    
    # DARK ENERGY RELATED CONSTANTS
    DARK_ENERGY_CONSTANTS = {
        'hubble_constant': 2.197e-18,              # H₀ in s⁻¹ (~70 km/s/Mpc)
        'cosmological_constant': 1.1056e-52,      # Λ in m⁻² (dark energy density parameter)
        'dark_energy_density': 5.96e-27,          # ρ_Λ in kg/m³ (dark energy density)
        'critical_density': 8.62e-27,             # ρ_c in kg/m³ (critical density of universe)
        'omega_lambda': 0.6889,                   # Ω_Λ (dark energy density parameter)
        'equation_of_state': -1.0,                # w (dark energy equation of state parameter)
        'planck_time': 5.391247e-44,              # t_P in seconds (Planck time)
        'planck_length': 1.616255e-35,            # l_P in meters (Planck length)
        'vacuum_energy_scale': 2.4e-3,            # eV (theoretical vacuum energy scale)
        'quintessence_mass': 3.16e-33,            # eV (hypothetical quintessence field mass)
        'dark_energy_scale': 0.0024,              # eV (characteristic dark energy scale)
        'acceleration_scale': 1.2e-10             # m/s² (cosmic acceleration scale)
    }
    
    # Specific tolerances for each constant (adjusted to their magnitudes)
    CONSTANT_TOLERANCES = {
        'hubble_constant': [1e-19, 1e-20, 1e-21, 1e-22, 1e-23, 1e-24],
        'cosmological_constant': [1e-53, 1e-54, 1e-55, 1e-56, 1e-57, 1e-58],
        'dark_energy_density': [1e-28, 1e-29, 1e-30, 1e-31, 1e-32, 1e-33],
        'critical_density': [1e-28, 1e-29, 1e-30, 1e-31, 1e-32, 1e-33],
        'omega_lambda': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        'equation_of_state': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        'planck_time': [1e-45, 1e-46, 1e-47, 1e-48, 1e-49, 1e-50],
        'planck_length': [1e-36, 1e-37, 1e-38, 1e-39, 1e-40, 1e-41],
        'vacuum_energy_scale': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
        'quintessence_mass': [1e-34, 1e-35, 1e-36, 1e-37, 1e-38, 1e-39],
        'dark_energy_scale': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
        'acceleration_scale': [1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]
    }
    
    # Control constants for statistical validation
    CONTROL_CONSTANTS = {
        'random_hubble': 2.2e-18,
        'random_lambda': 1.1e-52,
        'random_density': 6.0e-27,
        'golden_ratio': (np.sqrt(5) - 1) / 2,
        'pi_cosmic': np.pi / 1e18,
        'e_cosmic': np.e / 1e18
    }
    
    # CRITERIA FOR SIGNIFICANT RESONANCE
    SIGNIFICANCE_CRITERIA = {
        'min_resonances': 10,           # Minimum number of resonances
        'min_significance_factor': 2.0, # Minimum significance factor (2x expected)
        'max_p_value': 0.01,           # Maximum p-value for significance
        'min_chi2_stat': 6.635         # χ² critical for p < 0.01
    }
    
    def __init__(self, zeros_file: str, results_dir: str = "zvt_dark_energy_results", 
                 cache_file: str = "zeta_zeros_cache_dark_energy.pkl", 
                 state_file: str = "session_state_dark_energy.json",
                 increment: int = 50000):
        """
        Initialize the dark energy resonance hunter
        
        Args:
            zeros_file: Path to the file with zeta function zeros
            results_dir: Directory to save results
            cache_file: Cache file for loaded zeros
            state_file: State file for resumption
            increment: Batch size for processing
        """
        self.zeros_file = zeros_file
        self.results_dir = results_dir
        self.cache_file = cache_file
        self.state_file = state_file
        self.increment = increment
        self.all_zeros = []
        self.session_state = SessionState()
        self.shutdown_requested = False
        
        # Set up directories
        os.makedirs(results_dir, exist_ok=True)
        
        # Set up signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Suppress warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        logger.info(f"ZVT Dark Energy Hunter initialized")
        logger.info(f"Zeros file: {zeros_file}")
        logger.info(f"Results directory: {results_dir}")
    
    def _signal_handler(self, signum, frame):
        """Handler for interrupt signals"""
        self.shutdown_requested = True
        logger.info(f"🛑 Shutdown requested. Saving state and finalizing current batch...")
    
    def save_session_state(self):
        """Save session state for resumption"""
        try:
            state_data = {
                'last_processed_index': self.session_state.last_processed_index,
                'best_resonances': {k: v.to_dict() for k, v in self.session_state.best_resonances.items()},
                'start_time': self.session_state.start_time,
                'total_zeros': self.session_state.total_zeros
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info(f"💾 Session state saved: {self.state_file}")
        except Exception as e:
            logger.error(f"❌ Error saving state: {e}")
    
    def load_session_state(self):
        """Load session state for resumption"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                self.session_state.last_processed_index = state_data.get('last_processed_index', 0)
                self.session_state.start_time = state_data.get('start_time', time.time())
                self.session_state.total_zeros = state_data.get('total_zeros', 0)
                
                # Rebuild Resonance objects
                for name, res_data in state_data.get('best_resonances', {}).items():
                    self.session_state.best_resonances[name] = Resonance(**res_data)
                
                logger.info(f"📂 Session state loaded: last index {self.session_state.last_processed_index:,}")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Error loading state: {e}")
                return False
        return False
    
    def load_zeros_from_file(self, start_index: int = 0) -> List[Tuple[int, float]]:
        """Load zeros from file with progress indicator and resumption"""
        zeros = []
        try:
            logger.info(f"📂 Loading zeros from file: {self.zeros_file}")
            logger.info(f"🔄 Resuming from index {start_index:,}")
            
            # First, count total lines
            logger.info("📊 Counting file lines...")
            with open(self.zeros_file, 'r') as f:
                total_lines = sum(1 for line in f if line.strip())
            logger.info(f"📊 Total lines found: {total_lines:,}")
            
            with open(self.zeros_file, 'r') as f:
                # Skip already processed lines
                for _ in range(start_index):
                    next(f, None)
                
                progress_counter = 0
                for line_num, line in enumerate(f, start=start_index + 1):
                    if self.shutdown_requested:
                        break
                    line = line.strip()
                    if line:
                        try:
                            zero = float(line)
                            zeros.append((line_num, zero))
                            progress_counter += 1
                            
                            # Show progress every 100,000 zeros
                            if progress_counter % 100000 == 0:
                                percent = ((start_index + progress_counter) / total_lines) * 100
                                logger.info(f"📈 Loaded {start_index + progress_counter:,} zeros ({percent:.1f}%)")
                                
                        except ValueError:
                            logger.warning(f"⚠️ Invalid line {line_num}: '{line}'")
                            continue
            
            logger.info(f"✅ {len(zeros):,} zeros loaded (total: {start_index + len(zeros):,} of {total_lines:,})")
            return zeros
        except Exception as e:
            logger.error(f"❌ Error reading file: {e}")
            return []
    
    def save_enhanced_cache(self, zeros_list, backup=True):
        """Save zeros to cache with backup option"""
        try:
            if backup and os.path.exists(self.cache_file):
                backup_file = f"{self.cache_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.cache_file, backup_file)
                logger.info(f"📦 Cache backup created: {backup_file}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(zeros_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"💾 Cache saved: {len(zeros_list)} zeros")
        except Exception as e:
            logger.error(f"❌ Error saving cache: {e}")
    
    def load_enhanced_cache(self, force_reload=False):
        """Load zeros from cache or file"""
        if not force_reload and os.path.exists(self.cache_file):
            try:
                logger.info(f"🔍 Checking existing cache...")
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        logger.info(f"✅ Valid cache: {len(data):,} zeros loaded")
                        return data
            except Exception as e:
                logger.warning(f"⚠️ Invalid cache ({e}), loading from file...")
        
        # Load from file
        start_index = self.session_state.last_processed_index
        zeros = self.load_zeros_from_file(start_index)
        if zeros:
            self.save_enhanced_cache(zeros)
        return zeros
    
    def find_resonances_for_constant(self, zeros_batch: List[Tuple[int, float]], constant_name: str, 
                                 constant_value: float, tolerance: float) -> List[Resonance]:
        """Find resonances for a specific constant (optimized)"""
        resonances = []
        constant_val = float(constant_value)  # Convert once for better performance
        
        for n, gamma in zeros_batch:
            mod_val = gamma % constant_val
            min_distance = min(mod_val, constant_val - mod_val)
            if min_distance < tolerance:
                resonances.append(Resonance(
                    zero_index=n,
                    gamma=gamma,
                    quality=min_distance,
                    tolerance=tolerance,
                    constant_name=constant_name,
                    constant_value=constant_val
                ))
        return resonances
    
    def find_multi_tolerance_resonances(self, zeros_batch: List[Tuple[int, float]], 
                                      constants_dict: Dict[str, float] = None) -> Dict[str, Dict[float, List[Resonance]]]:
        """Find resonances at multiple tolerance levels (optimized)"""
        if constants_dict is None:
            constants_dict = self.DARK_ENERGY_CONSTANTS
            
        all_results = {}
        
        # Use ThreadPoolExecutor to parallelize by constant
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(constants_dict)) as executor:
            futures = {}
            
            for const_name, const_value in constants_dict.items():
                tolerances_to_use = self.CONSTANT_TOLERANCES.get(const_name, [1e-4, 1e-5, 1e-6])
                
                # Submit task for each tolerance
                for tolerance in tolerances_to_use:
                    future = executor.submit(
                        self.find_resonances_for_constant, 
                        zeros_batch, const_name, const_value, tolerance
                    )
                    futures[(const_name, tolerance)] = future
            
            # Collect results
            for (const_name, tolerance), future in futures.items():
                try:
                    resonances = future.result()
                    
                    if const_name not in all_results:
                        all_results[const_name] = {}
                    
                    all_results[const_name][tolerance] = resonances
                except Exception as e:
                    logger.error(f"Error processing {const_name} with tolerance {tolerance}: {e}")
        
        return all_results
    
    def enhanced_statistical_analysis(self, zeros_batch: List[Tuple[int, float]], 
                                    resonances: List[Resonance]) -> Dict[str, Any]:
        """Perform enhanced statistical analysis (simplified)"""
        if len(zeros_batch) == 0 or len(resonances) == 0:
            return {}
        
        total_zeros = len(zeros_batch)
        resonant_count = len(resonances)
        constant_value = resonances[0].constant_value
        tolerance = resonances[0].tolerance
        
        expected_random = total_zeros * (2 * tolerance / constant_value)
        
        basic_stats = {
            'total_zeros': total_zeros,
            'resonant_count': resonant_count,
            'expected_random': expected_random,
            'resonance_rate': resonant_count / total_zeros,
            'significance_factor': resonant_count / expected_random if expected_random > 0 else float('inf')
        }
        
        return basic_stats
    
    def analyze_batch_optimized(self, zeros_batch: List[Tuple[int, float]], batch_num: int) -> Dict[str, Any]:
        """Analyze a batch of zeros in an optimized way"""
        logger.info(f"\n🔬 BATCH #{batch_num}: {len(zeros_batch):,} zeros")
        
        # Analysis of dark energy constants
        constants_results = self.find_multi_tolerance_resonances(zeros_batch, self.DARK_ENERGY_CONSTANTS)
        
        best_resonances = {}
        
        logger.info(f"\n🌌 DARK ENERGY CONSTANTS ANALYSIS:")
        logger.info("| Constant              | Value         | Tolerance | Resonances | Rate (%) | Significance |")
        logger.info("|-----------------------|---------------|-----------|------------|----------|--------------|")
        
        for constant_name, constant_value in self.DARK_ENERGY_CONSTANTS.items():
            tolerances_to_check = self.CONSTANT_TOLERANCES.get(constant_name, [1e-4, 1e-5, 1e-6])
            
            for tolerance in tolerances_to_check[:3]:  # Check only the top 3 tolerances
                try:
                    if tolerance in constants_results[constant_name]:
                        resonances = constants_results[constant_name][tolerance]
                        count = len(resonances)
                        rate = count / len(zeros_batch) * 100 if len(zeros_batch) > 0 else 0
                        
                        stats_result = self.enhanced_statistical_analysis(zeros_batch, resonances)
                        sig_factor = stats_result.get('significance_factor', 0)
                        
                        # Show significance status
                        sig_marker = "🚨" if sig_factor > 2.0 else "  "
                        logger.info(f"|{sig_marker}{constant_name:21s} | {constant_value:.12e} | {tolerance:8.0e} | {count:10d} | {rate:8.3f} | {sig_factor:8.2f}x |")
                        
                        # Save best resonance
                        if resonances:
                            current_best = min(resonances, key=lambda x: x.quality)
                            if constant_name not in best_resonances or current_best.quality < best_resonances[constant_name].quality:
                                best_resonances[constant_name] = current_best
                        
                        # Register significant resonance
                        if sig_factor > 2.0:
                            logger.info(f"\n🚨 SIGNIFICANT DARK ENERGY RESONANCE: {constant_name.upper()}!")
                            logger.info(f"   Zero #{current_best.zero_index:,} → γ={current_best.gamma:.12f}")
                            logger.info(f"   Quality: {current_best.quality:.6e} (tolerance: {tolerance:.0e})")
                    
                except Exception as e:
                    logger.error(f"❌ Error in analysis of {constant_name}: {e}")
                    continue
        
        return {
            'batch_number': batch_num,
            'zeros_analyzed': len(zeros_batch),
            'best_resonances': best_resonances,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_final_report(self):
        """Generate a complete final report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.results_dir, f"Dark_Energy_Report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ZVT DARK ENERGY HUNTER - FINAL REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Zeros file: {self.zeros_file}\n")
            f.write(f"Total zeros processed: {self.session_state.last_processed_index:,}\n\n")
            
            f.write("DARK ENERGY CONSTANTS ANALYZED:\n")
            f.write("-" * 80 + "\n")
            for name, value in self.DARK_ENERGY_CONSTANTS.items():
                f.write(f"{name.replace('_', ' ').title()}: {value:.15e}\n")
            
            f.write("\nDESCRIPTION OF DARK ENERGY CONSTANTS:\n")
            f.write("-" * 80 + "\n")
            f.write("Hubble Constant (H₀): Rate of expansion of the universe\n")
            f.write("Cosmological Constant (Λ): Einstein's cosmological constant for dark energy\n")
            f.write("Dark Energy Density (ρ_Λ): Energy density of dark energy\n")
            f.write("Critical Density (ρ_c): Critical density for flat universe\n")
            f.write("Omega Lambda (Ω_Λ): Dark energy density parameter\n")
            f.write("Equation of State (w): Dark energy equation of state parameter\n")
            f.write("Planck Time (t_P): Fundamental time scale in quantum gravity\n")
            f.write("Planck Length (l_P): Fundamental length scale in quantum gravity\n")
            f.write("Vacuum Energy Scale: Theoretical scale of vacuum energy\n")
            f.write("Quintessence Mass: Hypothetical dark energy field mass\n")
            f.write("Dark Energy Scale: Characteristic energy scale of dark energy\n")
            f.write("Acceleration Scale: Scale of cosmic acceleration\n")
            
            f.write("\nRESULTS OF BEST DARK ENERGY RESONANCES:\n")
            f.write("-" * 80 + "\n")
            
            for constant_name, res in self.session_state.best_resonances.items():
                error_percent = (res.quality / self.DARK_ENERGY_CONSTANTS[constant_name]) * 100
                f.write(f"\n{constant_name.upper().replace('_', ' ')}:\n")
                f.write(f"  Zero #{res.zero_index:,}\n")
                f.write(f"  Gamma: {res.gamma:.15f}\n")
                f.write(f"  Quality: {res.quality:.15e}\n")
                f.write(f"  Relative error: {error_percent:.12f}%\n")
                f.write(f"  Tolerance: {res.tolerance:.0e}\n")
                f.write(f"  Estimated energy: {res.gamma/10:.6f} GeV\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("COSMOLOGICAL IMPLICATIONS:\n")
            f.write("-" * 80 + "\n")
            f.write("1. Dark energy represents ~68% of the universe's energy density\n")
            f.write("2. Responsible for the accelerating expansion of the universe\n")
            f.write("3. Connection with quantum field theory and vacuum energy\n")
            f.write("4. Potential link between microscopic quantum and macroscopic cosmic scales\n")
            f.write("5. Resonances may indicate deep mathematical structure in cosmology\n")
            
        logger.info(f"📊 Dark Energy report saved: {report_file}")
        return report_file
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations of the dark energy results"""
        if not self.session_state.session_results:
            logger.warning("No results to visualize")
            return
        
        # Prepare data for visualization
        constants_data = {constant: [] for constant in self.DARK_ENERGY_CONSTANTS.keys()}
        
        for result in self.session_state.session_results:
            for constant_name, res in result.get('best_resonances', {}).items():
                constants_data[constant_name].append({
                    'batch': result['batch_number'],
                    'quality': res.quality,
                    'zero_index': res.zero_index,
                    'gamma': res.gamma
                })
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        fig.suptitle('Dark Energy Resonance Analysis - Zeta Function Zeros', fontsize=16)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Plot for each dark energy constant
        plot_idx = 0
        for constant_name, data in constants_data.items():
            if plot_idx < len(axes) and data:
                ax = axes[plot_idx]
                
                batches = [d['batch'] for d in data]
                qualities = [d['quality'] for d in data]
                
                ax.plot(batches, qualities, 'o-', label=constant_name.replace('_', ' ').title(), 
                       color=plt.cm.viridis(plot_idx/len(constants_data)), linewidth=2, markersize=6)
                
                ax.set_yscale('log')
                ax.set_xlabel('Batch Number')
                ax.set_ylabel('Quality (log scale)')
                ax.set_title(f'{constant_name.replace("_", " ").title()}\nResonance Quality vs Batch')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = os.path.join(self.results_dir, f"dark_energy_visualizations_{timestamp}.png")
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Dark Energy visualizations saved: {viz_file}")
        
        # Close figure to free memory
        plt.close(fig)
        
        # Create summary comparison plot
        self.create_summary_comparison_plot()
    
    def create_summary_comparison_plot(self):
        """Create a summary comparison plot of all dark energy constants"""
        if not self.session_state.best_resonances:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dark Energy Constants - Summary Analysis', fontsize=16)
        
        constants = list(self.session_state.best_resonances.keys())
        qualities = [res.quality for res in self.session_state.best_resonances.values()]
        energies = [res.gamma/10 for res in self.session_state.best_resonances.values()]
        constant_values = [self.DARK_ENERGY_CONSTANTS[const] for const in constants]
        
        # 1. Best resonance qualities
        bars1 = ax1.bar(range(len(constants)), qualities, color=plt.cm.viridis(np.linspace(0, 1, len(constants))))
        ax1.set_yscale('log')
        ax1.set_ylabel('Quality (log scale)')
        ax1.set_title('Best Resonance Quality by Constant')
        ax1.set_xticks(range(len(constants)))
        ax1.set_xticklabels([c.replace('_', '\n') for c in constants], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Estimated energies
        bars2 = ax2.bar(range(len(constants)), energies, color=plt.cm.plasma(np.linspace(0, 1, len(constants))))
        ax2.set_ylabel('Estimated Energy (GeV)')
        ax2.set_title('Estimated Energies of Best Resonances')
        ax2.set_xticks(range(len(constants)))
        ax2.set_xticklabels([c.replace('_', '\n') for c in constants], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Constant magnitudes
        bars3 = ax3.bar(range(len(constants)), constant_values, color=plt.cm.magma(np.linspace(0, 1, len(constants))))
        ax3.set_yscale('log')
        ax3.set_ylabel('Constant Value (log scale)')
        ax3.set_title('Magnitudes of Dark Energy Constants')
        ax3.set_xticks(range(len(constants)))
        ax3.set_xticklabels([c.replace('_', '\n') for c in constants], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Relative errors
        relative_errors = [(res.quality / self.DARK_ENERGY_CONSTANTS[const]) * 100 
                          for const, res in self.session_state.best_resonances.items()]
        
        bars4 = ax4.bar(range(len(constants)), relative_errors, color=plt.cm.inferno(np.linspace(0, 1, len(constants))))
        ax4.set_yscale('log')
        ax4.set_ylabel('Relative Error (%)')
        ax4.set_title('Relative Errors of Best Resonances')
        ax4.set_xticks(range(len(constants)))
        ax4.set_xticklabels([c.replace('_', '\n') for c in constants], rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save summary plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.results_dir, f"dark_energy_summary_{timestamp}.png")
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Dark Energy summary plot saved: {summary_file}")
        
        # Close figure to free memory
        plt.close(fig)
    
    def run_analysis_optimized(self, force_reload=False):
        """Run complete dark energy analysis with resumption and optimized performance"""
        logger.info(f"🌌 ZVT DARK ENERGY HUNTER")
        logger.info(f"🔍 Searching for resonances with dark energy constants")
        logger.info("=" * 80)
        
        # Try to load session state
        session_loaded = self.load_session_state()
        
        if session_loaded:
            logger.info(f"🔄 Resuming previous session...")
        else:
            logger.info(f"🆕 Starting new session...")
        
        # Show dark energy constants
        logger.info(f"\n🌌 DARK ENERGY CONSTANTS TO ANALYZE:")
        logger.info("=" * 80)
        for constant_name, constant_value in self.DARK_ENERGY_CONSTANTS.items():
            tolerances_str = ", ".join([f"{t:.0e}" for t in self.CONSTANT_TOLERANCES.get(constant_name, [1e-4, 1e-5, 1e-6])[:3]])
            description = self._get_constant_description(constant_name)
            logger.info(f"🔬 {constant_name.upper().replace('_', ' ')}: {constant_value:.15e}")
            logger.info(f"   Description: {description}")
            logger.info(f"   Tolerances: {tolerances_str}")
            logger.info("")
        
        logger.info(f"📁 File: {self.zeros_file}")
        logger.info("🛑 Ctrl+C to stop (state will be saved)")
        logger.info("🚨 Significant dark energy resonances will be automatically highlighted")
        logger.info("=" * 80)
        
        # Load zeros
        self.all_zeros = self.load_enhanced_cache(force_reload=force_reload)
        current_count = len(self.all_zeros)
        
        if current_count == 0:
            logger.error("❌ No zeros loaded. Check the file.")
            return [], [], None
        
        # Update total zeros in state
        self.session_state.total_zeros = current_count
        
        logger.info(f"🎯 PROCESSING {current_count:,} ZEROS FOR DARK ENERGY ANALYSIS!")
        logger.info(f"📦 Batches of {self.increment:,} zeros each")
        logger.info(f"⏱️ Estimate: ~{(current_count//self.increment)} batches")
        
        batch_num = 1
        
        # Create progress bar
        pbar = tqdm(total=current_count, desc="🌌 Analyzing dark energy", unit="zeros")
        
        for i in range(0, current_count, self.increment):
            if self.shutdown_requested:
                break
            
            batch_start = i
            batch_end = min(i + self.increment, current_count)
            batch = self.all_zeros[batch_start:batch_end]
            
            # Progress indicator
            progress_percent = (batch_end / current_count) * 100
            logger.info(f"\n🔬 DARK ENERGY BATCH #{batch_num}: Zeros {batch_start:,} to {batch_end:,} ({progress_percent:.1f}% complete)")
            start_time = time.time()
            
            batch_result = self.analyze_batch_optimized(batch, batch_num)
            
            # Update information
            elapsed = time.time() - start_time
            batch_result['batch_time'] = elapsed
            batch_result['progress_percent'] = progress_percent
            
            # Update best global resonances
            if batch_result['best_resonances']:
                for constant_name, res in batch_result['best_resonances'].items():
                    if constant_name not in self.session_state.best_resonances or res.quality < self.session_state.best_resonances[constant_name].quality:
                        self.session_state.best_resonances[constant_name] = res
                        logger.info(f"    🎯 NEW GLOBAL BEST for {constant_name.upper()}!")
                        logger.info(f"    Zero #{res.zero_index:,} → γ={res.gamma:.12f}, quality={res.quality:.6e}")
            
            # Calculate performance statistics
            zeros_per_sec = len(batch) / elapsed if elapsed > 0 else 0
            remaining_zeros = current_count - batch_end
            eta_seconds = remaining_zeros / zeros_per_sec if zeros_per_sec > 0 else 0
            eta_hours = eta_seconds / 3600
            
            logger.info(f"⏱️ Batch processed in {elapsed:.1f}s ({zeros_per_sec:,.0f} zeros/s)")
            if eta_hours > 0:
                logger.info(f"📈 ETA for completion: {eta_hours:.1f} hours")
            
            self.session_state.session_results.append(batch_result)
            self.session_state.last_processed_index = batch_end
            
            # Save state periodically
            if batch_num % 10 == 0:  # Every 10 batches
                self.save_session_state()
            
            # Update progress bar
            pbar.update(len(batch))
            
            batch_num += 1
        
        # Close progress bar
        pbar.close()
        
        # Save final state
        self.save_session_state()
        
        # Generate final report
        self.generate_final_report()
        
        # Create visualizations
        self.create_comprehensive_visualizations()
        
        # Show best global dark energy resonances
        if self.session_state.best_resonances:
            logger.info(f"\n" + "🌌" * 60)
            logger.info(f"🏆 BEST DARK ENERGY RESONANCES FOUND 🏆")
            logger.info(f"🌌" * 60)
            logger.info("| Constant              | Zero #        | Gamma                | Quality        | Energy (GeV) |")
            logger.info("|-----------------------|---------------|----------------------|----------------|--------------|")
            for constant_name, res in self.session_state.best_resonances.items():
                logger.info(f"| {constant_name:21s} | {res.zero_index:13,} | {res.gamma:20.15f} | {res.quality:.6e} | {res.gamma/10:12.6f} |")
            
            logger.info(f"\n💎 DETAILS OF BEST DARK ENERGY RESONANCES:")
            for constant_name, res in self.session_state.best_resonances.items():
                error_percent = (res.quality / self.DARK_ENERGY_CONSTANTS[constant_name]) * 100
                description = self._get_constant_description(constant_name)
                logger.info(f"\n🔬 {constant_name.upper().replace('_', ' ')}:")
                logger.info(f"   Description: {description}")
                logger.info(f"   Zero #{res.zero_index:,} (γ = {res.gamma:.15f})")
                logger.info(f"   Quality: {res.quality:.15e}")
                logger.info(f"   Relative error: {error_percent:.12f}%")
                logger.info(f"   Tolerance: {res.tolerance:.0e}")
                logger.info(f"   Estimated energy: {res.gamma/10:.6f} GeV")
                logger.info(f"   Cosmological significance: {self._get_cosmological_significance(constant_name)}")
            logger.info(f"🌌" * 60)
        
        # Final summary
        total_processed = self.session_state.last_processed_index
        elapsed_time = time.time() - self.session_state.start_time
        
        logger.info(f"\n" + "="*60)
        logger.info(f"📊 FINAL DARK ENERGY ANALYSIS SUMMARY")
        logger.info(f"="*60)
        logger.info(f"📈 Total zeros processed: {total_processed:,} of {current_count:,}")
        logger.info(f"📈 Percentage completed: {(total_processed/current_count)*100:.1f}%")
        logger.info(f"📈 Batches processed: {len(self.session_state.session_results)}")
        logger.info(f"📈 Total time: {elapsed_time:.1f}s ({elapsed_time/3600:.1f}h)")
        logger.info(f"📈 Average speed: {total_processed/elapsed_time:,.0f} zeros/s")
        logger.info(f"🌌 Dark energy constants analyzed: {len(self.DARK_ENERGY_CONSTANTS)}")
        logger.info(f"🎯 Best resonances found: {len(self.session_state.best_resonances)}")
        logger.info(f"="*60)
        
        return self.all_zeros, self.session_state.session_results, self.session_state.best_resonances
    
    def _get_constant_description(self, constant_name: str) -> str:
        """Get description for a dark energy constant"""
        descriptions = {
            'hubble_constant': 'Rate of cosmic expansion (H₀)',
            'cosmological_constant': 'Einstein\'s cosmological constant (Λ)',
            'dark_energy_density': 'Energy density of dark energy (ρ_Λ)',
            'critical_density': 'Critical density of the universe (ρ_c)',
            'omega_lambda': 'Dark energy density parameter (Ω_Λ)',
            'equation_of_state': 'Dark energy equation of state (w)',
            'planck_time': 'Fundamental quantum time scale (t_P)',
            'planck_length': 'Fundamental quantum length scale (l_P)',
            'vacuum_energy_scale': 'Theoretical vacuum energy scale',
            'quintessence_mass': 'Hypothetical dark energy field mass',
            'dark_energy_scale': 'Characteristic dark energy scale',
            'acceleration_scale': 'Scale of cosmic acceleration'
        }
        return descriptions.get(constant_name, 'Unknown dark energy constant')
    
    def _get_cosmological_significance(self, constant_name: str) -> str:
        """Get cosmological significance for a constant"""
        significance = {
            'hubble_constant': 'Determines age and size of observable universe',
            'cosmological_constant': 'Drives accelerating expansion of universe',
            'dark_energy_density': 'Dominates total energy budget (~68% of universe)',
            'critical_density': 'Determines geometry of spacetime',
            'omega_lambda': 'Controls fate of universe expansion',
            'equation_of_state': 'Distinguishes dark energy models',
            'planck_time': 'Quantum gravity becomes important',
            'planck_length': 'Spacetime discreteness may emerge',
            'vacuum_energy_scale': 'Quantum field contributions to cosmology',
            'quintessence_mass': 'Dynamic dark energy field properties',
            'dark_energy_scale': 'Fundamental energy scale of cosmic acceleration',
            'acceleration_scale': 'Observable cosmic acceleration magnitude'
        }
        return significance.get(constant_name, 'General dark energy significance')

# Main execution
if __name__ == "__main__":
    zeros_file = os.path.expanduser("~/zeta/zero.txt")  # Path to the zeros file
    
    # Check if should force reload
    force_reload = len(sys.argv) > 1 and sys.argv[1] == '--force-reload'
    
    try:
        logger.info("🌌 Starting ZVT Dark Energy Hunter")
        hunter = ZetaDarkEnergyHunter(zeros_file)
        zeros, results, best = hunter.run_analysis_optimized(force_reload=force_reload)
        
        if zeros and len(zeros) > 0:
            logger.info(f"\n🎯 Dark Energy Analysis Complete!")
            logger.info(f"📁 Results in: {hunter.results_dir}/")
            logger.info(f"💾 Cache: {hunter.cache_file}")
            logger.info(f"🔄 State: {hunter.state_file}")
            logger.info(f"\n🌌 COSMOLOGICAL INSIGHTS:")
            logger.info("- Dark energy represents the dominant component of our universe")
            logger.info("- These resonances may reveal deep connections between quantum and cosmic scales")
            logger.info("- Mathematical patterns in zeta zeros could reflect fundamental physics")
            logger.info("- Results may guide future dark energy research and observations")
    except KeyboardInterrupt:
        logger.info(f"\n⏹️ Analysis interrupted. State saved for resumption.")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info(f"\n🌌 Dark Energy session completed!")
