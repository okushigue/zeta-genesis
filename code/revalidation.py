#!/usr/bin/env python3
"""
revalidation.py - Enhanced version for quantum circuit validation
Compatible with Qiskit 2.1.1+ with improved structure and functionality
"""
import argparse
import json
import os
import time
import math
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Corrected imports for Qiskit 2.1.1
try:
    from qiskit import QuantumCircuit, transpile, qpy
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError as e:
    print("Error importing Qiskit:", e)
    print("Execute: pip install qiskit-aer qiskit-ibm-runtime")
    raise

# Enhanced logging configuration
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging with file and console handlers"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

@dataclass
class ExperimentConfig:
    """Configuration for quantum experiments"""
    shots: int
    seed: int
    p0: float
    target_states: List[str]
    backends: List[str]
    qpy_path: Optional[str] = None
    max_retries: int = 3
    timeout: int = 3600
    log_level: str = "INFO"
    log_file: Optional[str] = None
    output_dir: str = "results"
    parallel_execution: bool = True
    optimization_level: int = 1
    grover_iterations: int = 1

@dataclass
class ExecutionResult:
    """Results from quantum circuit execution"""
    backend: str
    circuit_index: int
    counts: Dict[str, int]
    summary: Dict[str, Union[float, int]]
    stats: Dict[str, float]
    execution_time: float
    timestamp: str = ""
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class StatisticalAnalysis:
    """Class for statistical calculations"""
    
    @staticmethod
    def se_binomial(p: float, n: int) -> float:
        """Calculate standard error for binomial distribution"""
        return math.sqrt(p * (1 - p) / n) if n > 0 else 0.0

    @staticmethod
    def z_two_sided(p_obs: float, p0: float, n: int) -> float:
        """Calculate z-score for two-sided test"""
        se = StatisticalAnalysis.se_binomial(p0, n)
        return (p_obs - p0) / se if se != 0 else float('inf')

    @staticmethod
    def p_from_z(z: float) -> float:
        """Calculate p-value from z-score"""
        return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    @staticmethod
    def ic95(p_obs: float, n: int) -> Tuple[float, float]:
        """Calculate 95% confidence interval"""
        se = StatisticalAnalysis.se_binomial(p_obs, n)
        z = 1.96
        return max(0.0, p_obs - z * se), min(1.0, p_obs + z * se)

    @staticmethod
    def effect_size(p_obs: float, p0: float) -> float:
        """Calculate effect size using arcsin transformation"""
        return 2 * math.asin(math.sqrt(p_obs)) - 2 * math.asin(math.sqrt(p0))

    @staticmethod
    def bh_fdr(pvals: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction"""
        n = len(pvals)
        if n == 0:
            return []
        
        indexed = sorted(enumerate(pvals), key=lambda x: x[1])
        adjusted = [0.0] * n
        
        for i, (idx, p) in enumerate(indexed, start=1):
            q = p * n / i
            adjusted[idx] = min(q, 1.0)
        
        return adjusted

    @staticmethod
    def bonferroni_correction(pvals: List[float], alpha: float = 0.05) -> float:
        """Calculate Bonferroni correction threshold"""
        return alpha / max(1, len(pvals))

class QuantumCircuitManager:
    """Class for managing quantum circuits"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def load_qpy(self, path: str) -> List[QuantumCircuit]:
        """Load quantum circuits from QPY file"""
        try:
            with open(path, 'rb') as fd:
                circuits = qpy.load(fd)
            self.logger.info(f"Loaded {len(circuits)} circuits from {path}")
            return circuits
        except Exception as e:
            self.logger.error(f"Failed to load QPY: {e}")
            raise
    
    def generate_grover_circuit(self, n_qubits: int, marked_states: List[str], 
                                iterations: int = 1) -> QuantumCircuit:
        """Generate Grover circuit for marked states with specified iterations"""
        if n_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
        
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Initial superposition
        for q in range(n_qubits):
            qc.h(q)
        
        # Grover iterations
        for _ in range(iterations):
            # Oracle for marked states
            qc.barrier()
            for state in marked_states:
                # Validate state
                if len(state) != n_qubits or not all(c in '01' for c in state):
                    raise ValueError(f"Invalid state {state} for {n_qubits} qubits")
                
                # Prepare the state
                for i, bit in enumerate(reversed(state)):
                    if bit == '0':
                        qc.x(i)
                
                # Marking with multi-controlled Z
                if n_qubits > 1:
                    qc.h(n_qubits-1)
                    qc.mcx(list(range(n_qubits-1)), n_qubits-1)
                    qc.h(n_qubits-1)
                else:
                    qc.z(0)
                
                # Undo preparation
                for i, bit in enumerate(reversed(state)):
                    if bit == '0':
                        qc.x(i)
            
            # Diffuser
            qc.barrier()
            for q in range(n_qubits):
                qc.h(q)
                qc.x(q)
            
            qc.h(n_qubits-1)
            qc.mcx(list(range(n_qubits-1)), n_qubits-1)
            qc.h(n_qubits-1)
            
            for q in range(n_qubits):
                qc.x(q)
                qc.h(q)
        
        qc.barrier()
        qc.measure(range(n_qubits), range(n_qubits))
        return qc

class BackendManager:
    """Class for managing quantum backends"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._backend_cache = {}
    
    def get_backend(self, backend_name: str):
        """Get backend for execution with caching"""
        if backend_name in self._backend_cache:
            return self._backend_cache[backend_name]
        
        try:
            if backend_name.startswith('ibm_'):
                service = QiskitRuntimeService()
                backend = service.backend(backend_name)
            elif backend_name == 'aer_simulator':
                backend = AerSimulator()
            else:
                raise ValueError(f"Unknown backend: {backend_name}")
            
            self._backend_cache[backend_name] = backend
            self.logger.info(f"Successfully initialized backend: {backend_name}")
            return backend
        except Exception as e:
            self.logger.error(f"Failed to get backend {backend_name}: {e}")
            raise

class JobMonitor:
    """Class for monitoring quantum jobs"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def monitor_job_simple(self, job, timeout: int = 300, poll_interval: int = 5) -> bool:
        """Simple job monitoring with configurable polling"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                status = job.status()
                if status.name == 'DONE':
                    self.logger.info("Job completed successfully")
                    return True
                elif status.name in ['CANCELLED', 'ERROR']:
                    self.logger.error(f"Job failed with status: {status.name}")
                    return False
                else:
                    elapsed = int(time.time() - start_time)
                    self.logger.info(f"Job status: {status.name} ({elapsed}s elapsed)")
                    time.sleep(poll_interval)
            except Exception as e:
                self.logger.warning(f"Error checking status: {e}")
                time.sleep(poll_interval)
        
        self.logger.error("Timeout exceeded")
        return False

class QuantumExperimentExecutor:
    """Class for executing quantum experiments"""
    
    def __init__(self, config: ExperimentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.backend_manager = BackendManager(logger)
        self.job_monitor = JobMonitor(logger)
        self.stats = StatisticalAnalysis()
    
    def run_on_backend(self, qc: QuantumCircuit, backend_name: str) -> ExecutionResult:
        """Execute circuit with retry and timeout"""
        start_time = time.time()
        circuit_index = getattr(qc, 'index', 0)
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.info(f"Attempt {attempt+1}/{self.config.max_retries} for circuit {circuit_index} on {backend_name}")
                
                # Get backend
                backend = self.backend_manager.get_backend(backend_name)
                
                # Transpilation
                t_qc = transpile(qc, backend=backend, seed_transpiler=self.config.seed, 
                                optimization_level=self.config.optimization_level)
                
                # Execution
                job = backend.run(t_qc, shots=self.config.shots)
                
                # Monitoring
                if not self.job_monitor.monitor_job_simple(job, self.config.timeout):
                    raise Exception("Job failed or timeout")
                
                result = job.result()
                counts = result.get_counts()
                
                execution_time = time.time() - start_time
                
                # Analysis
                summ = self.summarize_counts(counts, self.config.target_states)
                stats = self.stat_summary(summ['p_obs'], self.config.p0, summ['total_shots'])
                
                return ExecutionResult(
                    backend=backend_name,
                    circuit_index=circuit_index,
                    counts=counts,
                    summary=summ,
                    stats=stats,
                    execution_time=execution_time
                )
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    self.logger.error("All attempts failed")
                    return ExecutionResult(
                        backend=backend_name,
                        circuit_index=circuit_index,
                        counts={},
                        summary={'total_shots': 0, 'success_counts': 0, 'p_obs': 0.0, 'effect_size': 0.0},
                        stats={'z': 0.0, 'p_value': 1.0, 'ci95_low': 0.0, 'ci95_high': 0.0, 'standard_error': 0.0},
                        execution_time=time.time() - start_time,
                        error_message=str(e)
                    )
                time.sleep(5)
    
    def summarize_counts(self, counts: Dict[str, int], target_states: List[str]) -> Dict[str, Union[float, int]]:
        """Summarize counts dictionary"""
        total = sum(counts.values())
        succ = sum(counts.get(s, 0) for s in target_states)
        p_obs = succ / total if total > 0 else 0.0
        
        return {
            'total_shots': total,
            'success_counts': succ,
            'p_obs': p_obs,
            'effect_size': self.stats.effect_size(p_obs, self.config.p0)
        }
    
    def stat_summary(self, p_obs: float, p0: float, n: int) -> Dict[str, float]:
        """Generate statistical summary"""
        z = self.stats.z_two_sided(p_obs, p0, n)
        pval = self.stats.p_from_z(z)
        ci_low, ci_high = self.stats.ic95(p_obs, n)
        
        return {
            'z': z,
            'p_value': pval,
            'ci95_low': ci_low,
            'ci95_high': ci_high,
            'standard_error': self.stats.se_binomial(p_obs, n)
        }
    
    def execute_single_backend(self, backend_name: str, circuits: List[QuantumCircuit]) -> List[ExecutionResult]:
        """Execute circuits on a specific backend"""
        self.logger.info(f"Executing on backend: {backend_name}")
        
        results = []
        for i, qc in enumerate(circuits):
            # Add index to circuit for tracking
            qc.index = i
            result = self.run_on_backend(qc, backend_name)
            results.append(result)
            
            if result.error_message:
                self.logger.error(f"Circuit {i} on {backend_name} failed: {result.error_message}")
            else:
                self.logger.info(f"Circuit {i} on {backend_name}: p_obs={result.summary['p_obs']:.4f}, "
                               f"p_value={result.stats['p_value']:.4g}")
        
        return results
    
    def execute_all_backends(self, circuits: List[QuantumCircuit]) -> List[ExecutionResult]:
        """Execute circuits on all configured backends"""
        all_results = []
        
        if self.config.parallel_execution and len(self.config.backends) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=len(self.config.backends)) as executor:
                future_to_backend = {
                    executor.submit(self.execute_single_backend, backend, circuits): backend
                    for backend in self.config.backends
                }
                
                for future in as_completed(future_to_backend):
                    backend = future_to_backend[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        self.logger.error(f"Execution on backend {backend} failed: {e}")
        else:
            # Sequential execution
            for backend in self.config.backends:
                results = self.execute_single_backend(backend, circuits)
                all_results.extend(results)
        
        # Apply statistical corrections
        if all_results:
            pvals = [r.stats['p_value'] for r in all_results]
            bonf_alpha = self.stats.bonferroni_correction(pvals)
            fdr_adj = self.stats.bh_fdr(pvals)
            
            for i, result in enumerate(all_results):
                result.stats['p_value_bonferroni_threshold'] = bonf_alpha
                result.stats['p_value_fdr_adj'] = fdr_adj[i]
        
        return all_results

class ResultsManager:
    """Class for managing experiment results"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def print_results_summary(self, results: List[ExecutionResult]):
        """Print formatted results summary"""
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        for result in results:
            print(f"\nBackend: {result.backend}")
            print(f"Circuit: {result.circuit_index}")
            print(f"Timestamp: {result.timestamp}")
            
            if result.error_message:
                print(f"ERROR: {result.error_message}")
            else:
                print(f"Success rate: {result.summary['p_obs']:.4f} "
                      f"({result.summary['success_counts']}/{result.summary['total_shots']})")
                print(f"P-value: {result.stats['p_value']:.4g}")
                print(f"Adjusted p-value (FDR): {result.stats['p_value_fdr_adj']:.4g}")
                print(f"CI95: [{result.stats['ci95_low']:.4f}, {result.stats['ci95_high']:.4f}]")
                print(f"Effect size: {result.summary['effect_size']:.4f}")
            
            print(f"Execution time: {result.execution_time:.2f}s")
            print("-" * 40)
    
    def save_results(self, results: List[ExecutionResult], config: ExperimentConfig, output_path: str):
        """Save results to JSON file"""
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        final_results = {
            'meta': {
                'timestamp': datetime.now().isoformat(),
                'shots': config.shots,
                'seed': config.seed,
                'backends': config.backends,
                'p0': config.p0,
                'target_states': config.target_states,
                'total_executions': len(results),
                'grover_iterations': config.grover_iterations,
                'optimization_level': config.optimization_level
            },
            'results': [asdict(r) for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
        return output_path
    
    def generate_report(self, results: List[ExecutionResult], config: ExperimentConfig, output_path: str):
        """Generate a detailed text report"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("QUANTUM EXPERIMENT REPORT\n")
            f.write("="*50 + "\n\n")
            
            # Configuration
            f.write("EXPERIMENT CONFIGURATION\n")
            f.write("-"*30 + "\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Shots: {config.shots}\n")
            f.write(f"Seed: {config.seed}\n")
            f.write(f"Backends: {', '.join(config.backends)}\n")
            f.write(f"Target states: {', '.join(config.target_states)}\n")
            f.write(f"Baseline probability (p0): {config.p0}\n")
            f.write(f"Grover iterations: {config.grover_iterations}\n")
            f.write(f"Optimization level: {config.optimization_level}\n\n")
            
            # Results summary
            f.write("RESULTS SUMMARY\n")
            f.write("-"*30 + "\n")
            
            # Group results by backend
            backend_results = {}
            for r in results:
                if r.backend not in backend_results:
                    backend_results[r.backend] = []
                backend_results[r.backend].append(r)
            
            for backend, b_results in backend_results.items():
                f.write(f"\nBackend: {backend}\n")
                
                # Calculate aggregate statistics
                valid_results = [r for r in b_results if not r.error_message]
                if valid_results:
                    avg_p_obs = np.mean([r.summary['p_obs'] for r in valid_results])
                    avg_effect_size = np.mean([r.summary['effect_size'] for r in valid_results])
                    min_pval = min([r.stats['p_value'] for r in valid_results])
                    
                    f.write(f"  Circuits executed: {len(valid_results)}/{len(b_results)}\n")
                    f.write(f"  Average success rate: {avg_p_obs:.4f}\n")
                    f.write(f"  Average effect size: {avg_effect_size:.4f}\n")
                    f.write(f"  Minimum p-value: {min_pval:.4g}\n")
                else:
                    f.write(f"  All circuits failed\n")
                
                # Individual circuit results
                f.write("  Circuit details:\n")
                for r in b_results:
                    f.write(f"    Circuit {r.circuit_index}: ")
                    if r.error_message:
                        f.write(f"ERROR - {r.error_message}\n")
                    else:
                        f.write(f"p_obs={r.summary['p_obs']:.4f}, p_value={r.stats['p_value']:.4g}\n")
        
        self.logger.info(f"Report saved to {output_path}")
        return output_path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced quantum circuit validation script")
    parser.add_argument('--qpy', help='Path to QPY file with circuits')
    parser.add_argument('--shots', type=int, default=20000, help='Number of shots')
    parser.add_argument('--backends', type=str, default='aer_simulator', help='Comma-separated list of backends')
    parser.add_argument('--out', type=str, default='revalidation_results.json', help='Output JSON file')
    parser.add_argument('--report', type=str, default='revalidation_report.txt', help='Output report file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--target_states', type=str, default='0000,0001,0010,0011', help='Comma-separated target states')
    parser.add_argument('--p0', type=float, default=0.25, help='Baseline probability')
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum retry attempts')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--log_file', type=str, help='Log file path')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel execution')
    parser.add_argument('--optimization_level', type=int, default=1, help='Transpiler optimization level (0-3)')
    parser.add_argument('--grover_iterations', type=int, default=1, help='Number of Grover iterations')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Starting quantum experiment revalidation")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare paths
    output_path = os.path.join(args.output_dir, args.out)
    report_path = os.path.join(args.output_dir, args.report)
    
    # Configuration
    config = ExperimentConfig(
        shots=args.shots,
        seed=args.seed,
        p0=args.p0,
        target_states=[s.strip() for s in args.target_states.split(',') if s.strip()],
        backends=[b.strip() for b in args.backends.split(',') if b.strip()],
        qpy_path=args.qpy,
        max_retries=args.max_retries,
        timeout=args.timeout,
        log_level=args.log_level,
        log_file=args.log_file,
        output_dir=args.output_dir,
        parallel_execution=args.parallel,
        optimization_level=args.optimization_level,
        grover_iterations=args.grover_iterations
    )
    
    logger.info(f"Configuration: {config}")
    
    # Load or generate circuits
    circuit_manager = QuantumCircuitManager(logger)
    circuits = []
    
    if args.qpy:
        circuits = circuit_manager.load_qpy(args.qpy)
    else:
        n_qubits = len(config.target_states[0])
        circuits = [circuit_manager.generate_grover_circuit(n_qubits, config.target_states, config.grover_iterations)]
    
    logger.info(f"Processing {len(circuits)} circuits on {len(config.backends)} backends")
    
    # Execute experiments
    executor = QuantumExperimentExecutor(config, logger)
    results = executor.execute_all_backends(circuits)
    
    # Process and save results
    results_manager = ResultsManager(logger)
    results_manager.print_results_summary(results)
    results_manager.save_results(results, config, output_path)
    results_manager.generate_report(results, config, report_path)
    
    logger.info("Experiment completed successfully")

if __name__ == '__main__':
    main()
