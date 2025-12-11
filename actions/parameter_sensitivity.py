"""
Parameter Sensitivity Analysis Module for Oware AI Deep Learning Models

This module implements comprehensive hyperparameter sensitivity analysis including:
- Systematic parameter sweep testing
- Sensitivity coefficient calculations  
- Statistical analysis with confidence intervals
- Performance visualization and reporting
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agents import DQNSmall, DQNMedium, DQNLarge, RandomAgent, GreedyAgent, HeuristicAgent, MinimaxAgent
from owareEngine import OwareBoard
from training import DQNTrainer, TrainingConfig, create_training_config

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Matplotlib/Seaborn not available. Visualizations will be disabled.")

# Check PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ParameterSensitivityAnalyzer:
    """Analyzes sensitivity of DQN performance to hyperparameter changes."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join('output', 'sensitivity_analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define parameter ranges for sensitivity testing
        self.parameter_ranges = {
            'learning_rate': [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
            'epsilon_decay': [0.990, 0.995, 0.999, 0.9995, 0.9999],
            'discount_factor': [0.90, 0.95, 0.99, 0.995, 0.999],
            'buffer_size': [1000, 5000, 10000, 20000, 50000],
            'batch_size': [16, 32, 64, 128, 256]
        }
        
        # Base configuration for testing
        self.base_config = {
            'learning_rate': 1e-3,
            'epsilon_decay': 0.995,
            'discount_factor': 0.99,
            'buffer_size': 10000,
            'batch_size': 64,
            'total_episodes': 1000,  # Reduced for faster sensitivity testing
            'eval_interval': 200,
            'eval_episodes': 30,
            'checkpoint_interval': 500,
            'variant': 'standard'
        }
        
        self.results = {}
        self.sensitivity_coefficients = {}
    
    def run_parameter_sweep(self, agent_class, parameter_name: str, 
                          num_runs: int = 3, parallel: bool = True) -> Dict:
        """
        Run sensitivity analysis for a specific parameter.
        
        Args:
            agent_class: DQN agent class to test
            parameter_name: Name of parameter to sweep
            num_runs: Number of runs per parameter value for statistical significance
            parallel: Whether to run tests in parallel
            
        Returns:
            Dictionary with results for each parameter value
        """
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Cannot run parameter sensitivity analysis.")
            return {}
        
        print(f"\nRunning parameter sweep for {parameter_name} on {agent_class.__name__}")
        print(f"Testing {len(self.parameter_ranges[parameter_name])} values with {num_runs} runs each")
        
        parameter_values = self.parameter_ranges[parameter_name]
        results = {}
        
        # Create test configurations
        test_configs = []
        for value in parameter_values:
            for run_id in range(num_runs):
                config = self.base_config.copy()
                config[parameter_name] = value
                test_configs.append((value, run_id, config))
        
        if parallel and len(test_configs) > 1:
            results = self._run_parallel_tests(agent_class, parameter_name, test_configs)
        else:
            results = self._run_sequential_tests(agent_class, parameter_name, test_configs)
        
        # Calculate statistics for each parameter value
        processed_results = self._process_parameter_results(results, parameter_values, num_runs)
        
        # Calculate sensitivity coefficients
        sensitivity = self._calculate_sensitivity_coefficient(processed_results, parameter_name)
        
        # Save results
        self._save_parameter_results(agent_class.__name__, parameter_name, processed_results, sensitivity)
        
        return processed_results
    
    def _run_sequential_tests(self, agent_class, parameter_name: str, test_configs: List) -> Dict:
        """Run parameter tests sequentially."""
        results = defaultdict(list)
        
        for i, (value, run_id, config) in enumerate(test_configs):
            print(f"  Progress: {i+1}/{len(test_configs)} - {parameter_name}={value}, run {run_id+1}")
            
            # Create and run training
            training_config = create_training_config(**config)
            trainer = DQNTrainer(training_config)
            
            # Run training with unique session name
            session_name = f"sensitivity_{agent_class.__name__}_{parameter_name}_{value}_{run_id}"
            training_results = trainer.train(agent_class, session_name)
            
            # Extract key metrics
            metrics = self._extract_training_metrics(training_results)
            results[value].append(metrics)
        
        return results
    
    def _run_parallel_tests(self, agent_class, parameter_name: str, test_configs: List) -> Dict:
        """Run parameter tests in parallel (with thread safety)."""
        results = defaultdict(list)
        results_lock = threading.Lock()
        
        def run_single_test(value, run_id, config):
            try:
                # Create training config
                training_config = create_training_config(**config)
                trainer = DQNTrainer(training_config)
                
                # Run training
                session_name = f"sensitivity_{agent_class.__name__}_{parameter_name}_{value}_{run_id}"
                training_results = trainer.train(agent_class, session_name)
                
                # Extract metrics
                metrics = self._extract_training_metrics(training_results)
                
                with results_lock:
                    results[value].append(metrics)
                
                return True
            except Exception as e:
                print(f"Error in test {parameter_name}={value}, run {run_id}: {e}")
                return False
        
        # Use ThreadPoolExecutor for parallel execution
        max_workers = min(4, len(test_configs))  # Limit concurrent training
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for value, run_id, config in test_configs:
                future = executor.submit(run_single_test, value, run_id, config)
                futures.append((future, value, run_id))
            
            # Monitor progress
            completed = 0
            for future, value, run_id in futures:
                try:
                    success = future.result(timeout=300)  # 5-minute timeout per test
                    completed += 1
                    print(f"  Progress: {completed}/{len(test_configs)} - {parameter_name}={value}, run {run_id+1}")
                except Exception as e:
                    print(f"Test failed: {parameter_name}={value}, run {run_id} - {e}")
        
        return results
    
    def _extract_training_metrics(self, training_results: Dict) -> Dict:
        """Extract key performance metrics from training results."""
        metrics = {
            'final_win_rate': training_results.get('final_win_rate', 0.0),
            'best_win_rate': training_results.get('best_win_rate', 0.0),
            'episodes_completed': training_results.get('episodes_completed', 0),
            'total_time': training_results.get('total_time', 0.0),
            'training_efficiency': 0.0
        }
        
        # Calculate training efficiency (win rate per minute)
        if metrics['total_time'] > 0:
            metrics['training_efficiency'] = metrics['final_win_rate'] / (metrics['total_time'] / 60.0)
        
        # Extract opponent-specific win rates from final results
        final_results = training_results.get('final_results', {})
        for opponent in ['random', 'greedy', 'heuristic', 'minimax']:
            key = f'{opponent}_win_rate'
            metrics[key] = final_results.get(key, 0.0)
        
        return metrics
    
    def _process_parameter_results(self, results: Dict, parameter_values: List, num_runs: int) -> Dict:
        """Process raw results into statistical summaries."""
        processed = {}
        
        for value in parameter_values:
            if value not in results or len(results[value]) == 0:
                continue
            
            runs = results[value]
            value_stats = {}
            
            # Calculate statistics for each metric
            all_metrics = set()
            for run in runs:
                all_metrics.update(run.keys())
            
            for metric in all_metrics:
                metric_values = [run.get(metric, 0.0) for run in runs]
                
                if len(metric_values) > 0:
                    mean_val = np.mean(metric_values)
                    std_val = np.std(metric_values, ddof=1) if len(metric_values) > 1 else 0.0
                    
                    # Calculate 95% confidence interval
                    if len(metric_values) > 1:
                        ci_95 = stats.t.interval(0.95, len(metric_values)-1, 
                                               loc=mean_val, 
                                               scale=stats.sem(metric_values))
                    else:
                        ci_95 = (mean_val, mean_val)
                    
                    value_stats[metric] = {
                        'mean': mean_val,
                        'std': std_val,
                        'min': np.min(metric_values),
                        'max': np.max(metric_values),
                        'ci_95_lower': ci_95[0],
                        'ci_95_upper': ci_95[1],
                        'sample_size': len(metric_values),
                        'raw_values': metric_values
                    }
            
            processed[value] = value_stats
        
        return processed
    
    def _calculate_sensitivity_coefficient(self, results: Dict, parameter_name: str) -> Dict:
        """Calculate sensitivity coefficients for each metric."""
        if len(results) < 2:
            return {}
        
        parameter_values = sorted(results.keys())
        coefficients = {}
        
        # Get all metrics
        sample_result = next(iter(results.values()))
        metrics = list(sample_result.keys())
        
        for metric in metrics:
            metric_means = []
            param_vals = []
            
            for param_val in parameter_values:
                if metric in results[param_val]:
                    metric_means.append(results[param_val][metric]['mean'])
                    param_vals.append(param_val)
            
            if len(metric_means) >= 2:
                # Calculate normalized sensitivity coefficient
                # S = (dY/Y) / (dX/X) where Y is metric, X is parameter
                
                param_range = max(param_vals) - min(param_vals)
                metric_range = max(metric_means) - min(metric_means)
                param_mean = np.mean(param_vals)
                metric_mean = np.mean(metric_means)
                
                if param_mean > 0 and metric_mean > 0 and param_range > 0:
                    # Normalized sensitivity
                    sensitivity = (metric_range / metric_mean) / (param_range / param_mean)
                    coefficients[metric] = {
                        'sensitivity_coefficient': sensitivity,
                        'parameter_range': param_range,
                        'metric_range': metric_range,
                        'correlation': np.corrcoef(param_vals, metric_means)[0, 1] if len(param_vals) > 1 else 0.0
                    }
        
        return coefficients
    
    def _save_parameter_results(self, agent_name: str, parameter_name: str, 
                              results: Dict, sensitivity: Dict):
        """Save parameter sweep results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(self.output_dir, 
                                  f"{agent_name}_{parameter_name}_results_{timestamp}.json")
        
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for param_val, metrics in results.items():
            json_results[str(param_val)] = {}
            for metric, stats in metrics.items():
                json_results[str(param_val)][metric] = {
                    k: float(v) if isinstance(v, (np.float64, np.float32)) else 
                       int(v) if isinstance(v, (np.int64, np.int32)) else 
                       [float(x) if isinstance(x, (np.float64, np.float32)) else x for x in v] if isinstance(v, list) else v
                    for k, v in stats.items()
                }
        
        with open(results_file, 'w') as f:
            json.dump({
                'agent': agent_name,
                'parameter': parameter_name,
                'timestamp': timestamp,
                'results': json_results,
                'sensitivity_analysis': sensitivity
            }, f, indent=2)
        
        print(f"Results saved to: {results_file}")
    
    def run_comprehensive_analysis(self, agent_classes: List = None, 
                                 parameters: List[str] = None,
                                 num_runs: int = 3) -> Dict:
        """
        Run comprehensive sensitivity analysis across multiple agents and parameters.
        
        Args:
            agent_classes: List of agent classes to test (default: all DQN variants)
            parameters: List of parameters to test (default: all defined parameters)
            num_runs: Number of runs per parameter value
            
        Returns:
            Complete results dictionary
        """
        if agent_classes is None:
            agent_classes = [DQNSmall, DQNMedium, DQNLarge]
        
        if parameters is None:
            parameters = list(self.parameter_ranges.keys())
        
        print(f"\nStarting comprehensive parameter sensitivity analysis")
        print(f"Agents: {[cls.__name__ for cls in agent_classes]}")
        print(f"Parameters: {parameters}")
        print(f"Runs per parameter value: {num_runs}")
        print(f"Total tests: {len(agent_classes) * len(parameters) * sum(len(self.parameter_ranges[p]) for p in parameters) * num_runs}")
        
        all_results = {}
        
        for agent_class in agent_classes:
            agent_results = {}
            for parameter in parameters:
                print(f"\n{'='*60}")
                print(f"Testing {agent_class.__name__} - {parameter}")
                print(f"{'='*60}")
                
                param_results = self.run_parameter_sweep(agent_class, parameter, num_runs)
                agent_results[parameter] = param_results
                
                # Calculate and store sensitivity coefficients
                sensitivity = self._calculate_sensitivity_coefficient(param_results, parameter)
                self.sensitivity_coefficients[f"{agent_class.__name__}_{parameter}"] = sensitivity
            
            all_results[agent_class.__name__] = agent_results
        
        # Save comprehensive summary
        self._save_comprehensive_summary(all_results)
        
        return all_results
    
    def _save_comprehensive_summary(self, all_results: Dict):
        """Save comprehensive analysis summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.output_dir, f"comprehensive_sensitivity_analysis_{timestamp}.json")
        
        summary = {
            'timestamp': timestamp,
            'analysis_config': {
                'parameter_ranges': self.parameter_ranges,
                'base_config': self.base_config
            },
            'results': all_results,
            'sensitivity_coefficients': self.sensitivity_coefficients
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nComprehensive analysis summary saved to: {summary_file}")
    
    def generate_sensitivity_report(self, results_file: str = None) -> str:
        """Generate a formatted sensitivity analysis report."""
        if results_file and os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
            results = data.get('results', {})
            sensitivity_coeffs = data.get('sensitivity_coefficients', {})
        else:
            results = {}
            sensitivity_coeffs = self.sensitivity_coefficients
        
        report = []
        report.append("PARAMETER SENSITIVITY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Sensitivity coefficients summary
        if sensitivity_coeffs:
            report.append("SENSITIVITY COEFFICIENTS SUMMARY")
            report.append("-" * 40)
            report.append(f"{'Parameter':<20} {'Agent':<12} {'Metric':<20} {'Coefficient':<12}")
            report.append("-" * 64)
            
            for key, metrics in sensitivity_coeffs.items():
                agent_param = key.split('_', 1)
                if len(agent_param) == 2:
                    agent, parameter = agent_param
                    for metric, data in metrics.items():
                        coeff = data.get('sensitivity_coefficient', 0.0)
                        report.append(f"{parameter:<20} {agent:<12} {metric:<20} {coeff:<12.4f}")
            
            report.append("")
        
        # Parameter ranges tested
        report.append("PARAMETER RANGES TESTED")
        report.append("-" * 30)
        for param, values in self.parameter_ranges.items():
            report.append(f"{param}: {values}")
        report.append("")
        
        # Base configuration
        report.append("BASE CONFIGURATION")
        report.append("-" * 20)
        for key, value in self.base_config.items():
            report.append(f"{key}: {value}")
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = os.path.join(self.output_dir, f"sensitivity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"Sensitivity report saved to: {report_file}")
        return report_text


def run_quick_sensitivity_test():
    """Run a quick sensitivity test for demonstration."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot run sensitivity analysis.")
        return
    
    print("Running quick sensitivity analysis (reduced parameters for speed)...")
    
    analyzer = ParameterSensitivityAnalyzer()
    
    # Override with smaller ranges for quick testing
    analyzer.parameter_ranges = {
        'learning_rate': [5e-4, 1e-3, 2e-3],
        'epsilon_decay': [0.995, 0.999],
        'batch_size': [32, 64, 128]
    }
    
    # Reduce training episodes for speed
    analyzer.base_config['total_episodes'] = 200
    analyzer.base_config['eval_interval'] = 100
    analyzer.base_config['eval_episodes'] = 20
    
    # Test one agent with one parameter
    results = analyzer.run_parameter_sweep(DQNSmall, 'learning_rate', num_runs=2)
    
    # Generate report
    report = analyzer.generate_sensitivity_report()
    print("\nQuick sensitivity test completed!")
    print("Sample results:", results)
    
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_sensitivity_test()
    else:
        print("Usage:")
        print("  python parameter_sensitivity.py quick  - Run quick test")
        print("  Import this module to run full analysis")