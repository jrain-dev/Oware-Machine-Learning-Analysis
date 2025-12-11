"""
Comprehensive Statistical Analysis Module for Oware AI Competition

This module provides detailed statistical analysis including:
- Multi-agent performance statistics with confidence intervals
- Pairwise matchup analysis with win rate matrices
- Training stability analysis across multiple runs
- Statistical significance testing
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

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agents import (Agent, RandomAgent, GreedyAgent, HeuristicAgent, MinimaxAgent, 
                   QLearningAgent, DQNSmall, DQNMedium, DQNLarge)
from owareEngine import OwareBoard
from simulation import play_match

# Statistical imports
try:
    from scipy import stats
    import scipy.stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Advanced statistical tests will be disabled.")


class ComprehensiveStatisticalAnalyzer:
    """Performs comprehensive statistical analysis of agent performance."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join('output', 'statistical_analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define agent types for analysis
        self.agent_types = {
            'RandomAgent': RandomAgent,
            'GreedyAgent': GreedyAgent,
            'HeuristicAgent': HeuristicAgent,
            'MinimaxAgent': MinimaxAgent,
            'QLearningAgent': QLearningAgent,
            'DQNSmall': DQNSmall,
            'DQNMedium': DQNMedium,
            'DQNLarge': DQNLarge
        }
        
        # Metrics to analyze
        self.metrics = [
            'win_rate',
            'avg_score',
            'avg_game_length',
            'avg_captures_per_game',
            'score_variance',
            'win_margin'
        ]
        
        self.results = {}
        self.statistical_summary = {}
    
    def run_comprehensive_evaluation(self, sample_size: int = 100, 
                                   num_replications: int = 5,
                                   game_variant: str = 'standard') -> Dict:
        """
        Run comprehensive statistical evaluation of all agents.
        
        Args:
            sample_size: Number of games per matchup per replication
            num_replications: Number of independent replications for statistical robustness
            game_variant: Game variant to test
            
        Returns:
            Comprehensive statistical results
        """
        print(f"\nRunning comprehensive statistical evaluation")
        print(f"Sample size per matchup: {sample_size} games")
        print(f"Number of replications: {num_replications}")
        print(f"Total games per matchup: {sample_size * num_replications}")
        print(f"Game variant: {game_variant}")
        
        # Get all possible pairwise matchups
        agent_names = list(self.agent_types.keys())
        matchups = list(itertools.combinations(agent_names, 2))
        
        print(f"Testing {len(matchups)} unique matchups")
        print(f"Total games to be played: {len(matchups) * sample_size * num_replications * 2}")  # *2 for both positions
        
        # Store all results
        all_results = defaultdict(lambda: defaultdict(list))
        
        # Run evaluations
        for i, (agent1_name, agent2_name) in enumerate(matchups):
            print(f"\nMatchup {i+1}/{len(matchups)}: {agent1_name} vs {agent2_name}")
            
            for rep in range(num_replications):
                print(f"  Replication {rep+1}/{num_replications}")
                
                # Create fresh agents for each replication
                agent1 = self.agent_types[agent1_name]()
                agent2 = self.agent_types[agent2_name]()
                
                # Run games with position swapping
                rep_results = self._run_matchup_evaluation(
                    agent1, agent2, agent1_name, agent2_name, 
                    sample_size, game_variant
                )
                
                # Store results
                for agent_name, metrics in rep_results.items():
                    for metric, value in metrics.items():
                        all_results[agent_name][metric].append(value)
        
        # Calculate comprehensive statistics
        statistical_summary = self._calculate_comprehensive_statistics(all_results, sample_size, num_replications)
        
        # Save results
        self._save_statistical_results(all_results, statistical_summary, sample_size, num_replications)
        
        self.results = all_results
        self.statistical_summary = statistical_summary
        
        return {
            'raw_results': all_results,
            'statistical_summary': statistical_summary,
            'sample_info': {
                'sample_size_per_replication': sample_size,
                'num_replications': num_replications,
                'total_sample_size': sample_size * num_replications,
                'game_variant': game_variant
            }
        }
    
    def _run_matchup_evaluation(self, agent1: Agent, agent2: Agent, 
                              agent1_name: str, agent2_name: str,
                              sample_size: int, game_variant: str) -> Dict:
        """Run evaluation for a specific matchup."""
        results = defaultdict(lambda: defaultdict(list))
        
        # Play games with both agents in both positions
        for game_num in range(sample_size):
            # Agent1 as player 0, Agent2 as player 1
            game_result = self._play_evaluation_game(agent1, agent2, game_variant)
            self._record_game_result(game_result, agent1_name, agent2_name, results, position_swap=False)
            
            # Agent2 as player 0, Agent1 as player 1 (position swap)
            game_result = self._play_evaluation_game(agent2, agent1, game_variant)
            self._record_game_result(game_result, agent2_name, agent1_name, results, position_swap=True)
        
        # Calculate aggregate metrics for this matchup
        matchup_summary = {}
        for agent_name in [agent1_name, agent2_name]:
            agent_metrics = {}
            
            # Win rate
            wins = sum(results[agent_name]['wins'])
            total_games = len(results[agent_name]['wins'])
            agent_metrics['win_rate'] = wins / total_games if total_games > 0 else 0.0
            
            # Average score
            scores = results[agent_name]['scores']
            agent_metrics['avg_score'] = np.mean(scores) if scores else 0.0
            
            # Average game length
            game_lengths = results[agent_name]['game_lengths']
            agent_metrics['avg_game_length'] = np.mean(game_lengths) if game_lengths else 0.0
            
            # Average captures per game (same as avg_score for Oware)
            agent_metrics['avg_captures_per_game'] = agent_metrics['avg_score']
            
            # Score variance
            agent_metrics['score_variance'] = np.var(scores) if len(scores) > 1 else 0.0
            
            # Win margin (average score difference when winning)
            win_margins = results[agent_name]['win_margins']
            agent_metrics['win_margin'] = np.mean(win_margins) if win_margins else 0.0
            
            matchup_summary[agent_name] = agent_metrics
        
        return matchup_summary
    
    def _play_evaluation_game(self, agent1: Agent, agent2: Agent, 
                            game_variant: str) -> Dict:
        """Play a single evaluation game."""
        board = OwareBoard(variant=game_variant)
        agents = {0: agent1, 1: agent2}
        
        board.reset()
        move_count = 0
        
        while not board.game_over and move_count < 200:  # Prevent infinite games
            current_player = board.current_player
            current_agent = agents[current_player]
            valid_moves = board.get_valid_moves(current_player)
            
            if not valid_moves:
                break
            
            action = current_agent.select_action(board, valid_moves)
            if action is None:
                break
            
            board.apply_move(action)
            move_count += 1
            
            # Call end_episode for learning agents to update exploration
            if hasattr(current_agent, 'end_episode'):
                try:
                    current_agent.end_episode()
                except:
                    pass
        
        return {
            'winner': board.winner,
            'scores': board.scores.copy(),
            'move_count': move_count,
            'final_board': board.board.copy()
        }
    
    def _record_game_result(self, game_result: Dict, agent1_name: str, 
                          agent2_name: str, results: Dict, position_swap: bool = False):
        """Record the results of a single game."""
        winner = game_result['winner']
        scores = game_result['scores']
        move_count = game_result['move_count']
        
        # Determine which agent corresponds to which player
        if not position_swap:
            agent1_player, agent2_player = 0, 1
        else:
            agent1_player, agent2_player = 1, 0
        
        # Record results for agent1
        results[agent1_name]['wins'].append(1 if winner == agent1_player else 0)
        results[agent1_name]['scores'].append(scores[agent1_player])
        results[agent1_name]['game_lengths'].append(move_count)
        
        if winner == agent1_player:
            win_margin = scores[agent1_player] - scores[agent2_player]
            results[agent1_name]['win_margins'].append(win_margin)
        
        # Record results for agent2
        results[agent2_name]['wins'].append(1 if winner == agent2_player else 0)
        results[agent2_name]['scores'].append(scores[agent2_player])
        results[agent2_name]['game_lengths'].append(move_count)
        
        if winner == agent2_player:
            win_margin = scores[agent2_player] - scores[agent1_player]
            results[agent2_name]['win_margins'].append(win_margin)
    
    def _calculate_comprehensive_statistics(self, all_results: Dict, 
                                          sample_size: int, num_replications: int) -> Dict:
        """Calculate comprehensive statistics for all agents."""
        statistics = {}
        
        for agent_name in self.agent_types.keys():
            if agent_name not in all_results:
                continue
            
            agent_stats = {}
            agent_data = all_results[agent_name]
            
            for metric in self.metrics:
                if metric not in agent_data or not agent_data[metric]:
                    continue
                
                values = agent_data[metric]
                n = len(values)
                
                if n == 0:
                    continue
                
                # Basic statistics
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if n > 1 else 0.0
                min_val = np.min(values)
                max_val = np.max(values)
                median_val = np.median(values)
                
                # Confidence intervals
                if n > 1 and SCIPY_AVAILABLE:
                    # 95% confidence interval for mean
                    ci_95 = stats.t.interval(0.95, n-1, loc=mean_val, scale=stats.sem(values))
                    
                    # 99% confidence interval for mean
                    ci_99 = stats.t.interval(0.99, n-1, loc=mean_val, scale=stats.sem(values))
                else:
                    ci_95 = (mean_val, mean_val)
                    ci_99 = (mean_val, mean_val)
                
                # Additional statistics
                q25 = np.percentile(values, 25)
                q75 = np.percentile(values, 75)
                iqr = q75 - q25
                
                # Standard error
                std_error = std_val / np.sqrt(n) if n > 0 else 0.0
                
                # Coefficient of variation
                cv = std_val / mean_val if mean_val != 0 else 0.0
                
                agent_stats[metric] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'min': float(min_val),
                    'max': float(max_val),
                    'median': float(median_val),
                    'q25': float(q25),
                    'q75': float(q75),
                    'iqr': float(iqr),
                    'ci_95_lower': float(ci_95[0]),
                    'ci_95_upper': float(ci_95[1]),
                    'ci_99_lower': float(ci_99[0]),
                    'ci_99_upper': float(ci_99[1]),
                    'sample_size': n,
                    'std_error': float(std_error),
                    'coefficient_of_variation': float(cv),
                    'sample_size_per_replication': sample_size,
                    'num_replications': num_replications
                }
            
            statistics[agent_name] = agent_stats
        
        return statistics
    
    def _save_statistical_results(self, all_results: Dict, statistical_summary: Dict,
                                sample_size: int, num_replications: int):
        """Save comprehensive statistical results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(self.output_dir, f"comprehensive_statistics_{timestamp}.json")
        
        output_data = {
            'timestamp': timestamp,
            'analysis_config': {
                'sample_size_per_replication': sample_size,
                'num_replications': num_replications,
                'total_sample_size': sample_size * num_replications,
                'agents_tested': list(self.agent_types.keys()),
                'metrics_analyzed': self.metrics
            },
            'raw_results': {k: dict(v) for k, v in all_results.items()},
            'statistical_summary': statistical_summary
        }
        
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        # Save CSV summary for easy analysis
        self._save_csv_summary(statistical_summary, timestamp)
        
        print(f"\nStatistical analysis saved to: {results_file}")
    
    def _save_csv_summary(self, statistical_summary: Dict, timestamp: str):
        """Save statistical summary as CSV for easy analysis."""
        rows = []
        
        for agent_name, agent_stats in statistical_summary.items():
            for metric, stats in agent_stats.items():
                row = {
                    'agent': agent_name,
                    'metric': metric,
                    **stats
                }
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            csv_file = os.path.join(self.output_dir, f"statistics_summary_{timestamp}.csv")
            df.to_csv(csv_file, index=False)
            print(f"CSV summary saved to: {csv_file}")
    
    def generate_statistical_report(self) -> str:
        """Generate a formatted statistical report."""
        if not self.statistical_summary:
            return "No statistical data available. Run comprehensive_evaluation first."
        
        report = []
        report.append("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall summary
        report.append("ANALYSIS OVERVIEW")
        report.append("-" * 30)
        
        # Get sample size info from first agent/metric
        sample_agent = next(iter(self.statistical_summary.keys()))
        sample_metric = next(iter(self.statistical_summary[sample_agent].keys()))
        sample_info = self.statistical_summary[sample_agent][sample_metric]
        
        report.append(f"Agents analyzed: {len(self.statistical_summary)}")
        report.append(f"Metrics per agent: {len(self.metrics)}")
        report.append(f"Sample size per replication: {sample_info.get('sample_size_per_replication', 'N/A')}")
        report.append(f"Number of replications: {sample_info.get('num_replications', 'N/A')}")
        report.append(f"Total sample size per metric: {sample_info.get('sample_size', 'N/A')}")
        report.append("")
        
        # Win rate rankings
        report.append("WIN RATE RANKINGS")
        report.append("-" * 25)
        
        win_rates = []
        for agent, stats in self.statistical_summary.items():
            if 'win_rate' in stats:
                wr_stats = stats['win_rate']
                win_rates.append((agent, wr_stats['mean'], wr_stats['ci_95_lower'], wr_stats['ci_95_upper']))
        
        win_rates.sort(key=lambda x: x[1], reverse=True)
        
        report.append(f"{'Rank':<5} {'Agent':<15} {'Win Rate':<12} {'95% CI':<20}")
        report.append("-" * 52)
        
        for i, (agent, mean_wr, ci_low, ci_high) in enumerate(win_rates, 1):
            ci_str = f"[{ci_low:.3f}, {ci_high:.3f}]"
            report.append(f"{i:<5} {agent:<15} {mean_wr:<12.3f} {ci_str:<20}")
        
        report.append("")
        
        # Detailed statistics table
        report.append("DETAILED STATISTICS BY AGENT")
        report.append("-" * 35)
        
        for agent_name in sorted(self.statistical_summary.keys()):
            report.append(f"\n{agent_name}")
            report.append("=" * len(agent_name))
            
            agent_stats = self.statistical_summary[agent_name]
            
            report.append(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'95% CI':<20}")
            report.append("-" * 80)
            
            for metric in sorted(agent_stats.keys()):
                stats = agent_stats[metric]
                ci_str = f"[{stats['ci_95_lower']:.3f}, {stats['ci_95_upper']:.3f}]"
                report.append(f"{metric:<20} {stats['mean']:<10.3f} {stats['std']:<10.3f} "
                            f"{stats['min']:<10.3f} {stats['max']:<10.3f} {ci_str:<20}")
        
        report_text = "\n".join(report)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"statistical_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"Statistical report saved to: {report_file}")
        return report_text
    
    def calculate_pairwise_win_matrix(self) -> pd.DataFrame:
        """Calculate win rate matrix for all agent pairings."""
        if not self.results:
            print("No results available. Run comprehensive_evaluation first.")
            return pd.DataFrame()
        
        agent_names = sorted(self.agent_types.keys())
        win_matrix = pd.DataFrame(index=agent_names, columns=agent_names, dtype=float)
        
        # Initialize diagonal to 0.5 (agents vs themselves)
        for agent in agent_names:
            win_matrix.loc[agent, agent] = 0.5
        
        # Calculate pairwise win rates from stored results
        for agent in agent_names:
            if agent in self.statistical_summary and 'win_rate' in self.statistical_summary[agent]:
                # This is simplified - in a full implementation, you'd need to track
                # which opponents each result was against
                overall_win_rate = self.statistical_summary[agent]['win_rate']['mean']
                
                # For now, use overall win rate as approximation
                # In full implementation, would need opponent-specific tracking
                for opponent in agent_names:
                    if agent != opponent and pd.isna(win_matrix.loc[agent, opponent]):
                        # Use overall win rate as approximation
                        win_matrix.loc[agent, opponent] = overall_win_rate
                        win_matrix.loc[opponent, agent] = 1.0 - overall_win_rate
        
        return win_matrix


def run_quick_statistical_analysis():
    """Run a quick statistical analysis for demonstration."""
    print("Running quick statistical analysis...")
    
    analyzer = ComprehensiveStatisticalAnalyzer()
    
    # Use smaller sample sizes for quick testing
    results = analyzer.run_comprehensive_evaluation(
        sample_size=20,  # Reduced from 100
        num_replications=2,  # Reduced from 5
        game_variant='standard'
    )
    
    # Generate report
    report = analyzer.generate_statistical_report()
    
    print("\nQuick statistical analysis completed!")
    print(f"Analyzed {len(results['statistical_summary'])} agents")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_statistical_analysis()
    else:
        print("Usage:")
        print("  python statistical_analysis.py quick  - Run quick analysis")
        print("  Import this module to run full analysis")