"""
Advanced Visualization Module for Oware AI Analysis

This module creates comprehensive visualizations including:
- Parameter sensitivity response curves
- Agent performance heatmaps
- Training stability charts
- Statistical distribution plots
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Visualization imports with fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Visualizations will be disabled.")

try:
    from scipy import stats
    import scipy.stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class AdvancedVisualizationSuite:
    """Creates comprehensive visualizations for Oware AI analysis."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join('output', 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style preferences
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
            sns.set_palette("husl")
        
        self.colors = {
            'DQNLarge': '#2E86AB',
            'DQNMedium': '#A23B72', 
            'DQNSmall': '#F18F01',
            'MinimaxAgent': '#C73E1D',
            'HeuristicAgent': '#6A994E',
            'GreedyAgent': '#F2CC8F',
            'RandomAgent': '#81B29A',
            'QLearningAgent': '#3D5A80'
        }
    
    def create_parameter_sensitivity_plots(self, sensitivity_results: Dict, 
                                         save_individual: bool = True) -> str:
        """
        Create parameter sensitivity response curves showing how performance
        responds to hyperparameter changes.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot create visualizations.")
            return ""
        
        print("Creating parameter sensitivity visualization...")
        
        # Create a comprehensive figure with subplots
        n_params = len(sensitivity_results)
        n_agents = len(next(iter(sensitivity_results.values()))) if sensitivity_results else 0
        
        if n_params == 0 or n_agents == 0:
            print("No sensitivity data available for visualization.")
            return ""
        
        # Create figure with subplots for each parameter
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        param_names = list(sensitivity_results.keys())
        
        for idx, param_name in enumerate(param_names[:6]):  # Limit to 6 parameters
            ax = axes[idx]
            
            param_data = sensitivity_results[param_name]
            
            # Plot sensitivity curves for each agent
            for agent_name, agent_results in param_data.items():
                if not agent_results:
                    continue
                
                # Extract parameter values and corresponding win rates
                param_values = sorted(agent_results.keys())
                win_rates = []
                error_bars = []
                
                for pval in param_values:
                    if 'win_rate' in agent_results[pval]:
                        wr_stats = agent_results[pval]['win_rate']
                        win_rates.append(wr_stats['mean'])
                        
                        # Calculate error bar (95% CI width)
                        error = (wr_stats['ci_95_upper'] - wr_stats['ci_95_lower']) / 2
                        error_bars.append(error)
                    else:
                        win_rates.append(0.0)
                        error_bars.append(0.0)
                
                if win_rates:
                    color = self.colors.get(agent_name, f'C{hash(agent_name) % 10}')
                    
                    # Plot line with error bars
                    ax.errorbar(param_values, win_rates, yerr=error_bars,
                              label=agent_name, color=color, marker='o',
                              linewidth=2, markersize=6, capsize=4)
            
            ax.set_xlabel(param_name.replace('_', ' ').title())
            ax.set_ylabel('Win Rate')
            ax.set_title(f'Sensitivity to {param_name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set appropriate x-scale
            if param_name in ['learning_rate']:
                ax.set_xscale('log')
        
        # Remove unused subplots
        for idx in range(len(param_names), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        # Save comprehensive plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.output_dir, f"parameter_sensitivity_analysis_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Parameter sensitivity plot saved to: {plot_file}")
        
        # Create individual plots if requested
        if save_individual:
            self._create_individual_sensitivity_plots(sensitivity_results, timestamp)
        
        return plot_file
    
    def _create_individual_sensitivity_plots(self, sensitivity_results: Dict, timestamp: str):
        """Create individual plots for each parameter."""
        for param_name, param_data in sensitivity_results.items():
            plt.figure(figsize=(12, 8))
            
            for agent_name, agent_results in param_data.items():
                if not agent_results:
                    continue
                
                param_values = sorted(agent_results.keys())
                win_rates = []
                ci_lower = []
                ci_upper = []
                
                for pval in param_values:
                    if 'win_rate' in agent_results[pval]:
                        wr_stats = agent_results[pval]['win_rate']
                        win_rates.append(wr_stats['mean'])
                        ci_lower.append(wr_stats['ci_95_lower'])
                        ci_upper.append(wr_stats['ci_95_upper'])
                    else:
                        win_rates.append(0.0)
                        ci_lower.append(0.0)
                        ci_upper.append(0.0)
                
                if win_rates:
                    color = self.colors.get(agent_name, f'C{hash(agent_name) % 10}')
                    
                    # Plot line
                    plt.plot(param_values, win_rates, label=agent_name, 
                           color=color, marker='o', linewidth=3, markersize=8)
                    
                    # Plot confidence interval
                    plt.fill_between(param_values, ci_lower, ci_upper, 
                                   color=color, alpha=0.2)
            
            plt.xlabel(param_name.replace('_', ' ').title(), fontsize=14)
            plt.ylabel('Win Rate', fontsize=14)
            plt.title(f'Performance Sensitivity to {param_name.replace("_", " ").title()}', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            
            if param_name in ['learning_rate']:
                plt.xscale('log')
            
            individual_file = os.path.join(self.output_dir, 
                                         f"sensitivity_{param_name}_{timestamp}.png")
            plt.savefig(individual_file, dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_performance_heatmap(self, win_rate_matrix: pd.DataFrame = None, 
                                 statistical_results: Dict = None) -> str:
        """
        Create a heatmap showing win rates for all pairwise agent matchups.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot create heatmap.")
            return ""
        
        print("Creating agent performance heatmap...")
        
        # If no matrix provided, create from statistical results
        if win_rate_matrix is None and statistical_results:
            win_rate_matrix = self._create_win_rate_matrix(statistical_results)
        
        if win_rate_matrix is None or win_rate_matrix.empty:
            print("No win rate data available for heatmap.")
            return ""
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        # Create custom colormap
        mask = np.triu(np.ones_like(win_rate_matrix, dtype=bool))
        
        # Create the heatmap
        ax = sns.heatmap(win_rate_matrix, 
                        annot=True, 
                        fmt='.3f',
                        cmap='RdYlBu_r',
                        center=0.5,
                        square=True,
                        linewidths=0.5,
                        mask=mask,
                        cbar_kws={"shrink": .8, "label": "Win Rate"})
        
        plt.title('Agent Performance Heatmap\n(Win Rates for Pairwise Matchups)', 
                 fontsize=16, pad=20)
        plt.xlabel('Opponent Agent', fontsize=14)
        plt.ylabel('Player Agent', fontsize=14)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        heatmap_file = os.path.join(self.output_dir, f"performance_heatmap_{timestamp}.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance heatmap saved to: {heatmap_file}")
        return heatmap_file
    
    def _create_win_rate_matrix(self, statistical_results: Dict) -> pd.DataFrame:
        """Create win rate matrix from statistical results."""
        agents = list(statistical_results['statistical_summary'].keys())
        matrix = pd.DataFrame(index=agents, columns=agents, dtype=float)
        
        # Fill diagonal with 0.5 (self vs self)
        for agent in agents:
            matrix.loc[agent, agent] = 0.5
        
        # Use overall win rates as approximation for pairwise matchups
        # In a full implementation, this would be calculated from actual pairwise games
        for i, agent1 in enumerate(agents):
            agent1_stats = statistical_results['statistical_summary'][agent1]
            agent1_wr = agent1_stats.get('win_rate', {}).get('mean', 0.5)
            
            for j, agent2 in enumerate(agents):
                if i != j:
                    agent2_stats = statistical_results['statistical_summary'][agent2]
                    agent2_wr = agent2_stats.get('win_rate', {}).get('mean', 0.5)
                    
                    # Approximate pairwise win rate based on relative strengths
                    if agent1_wr + agent2_wr > 0:
                        estimated_wr = agent1_wr / (agent1_wr + agent2_wr)
                    else:
                        estimated_wr = 0.5
                    
                    matrix.loc[agent1, agent2] = estimated_wr
        
        return matrix
    
    def create_stability_comparison_chart(self, training_data: Dict = None, 
                                        dqn_results: Dict = None) -> str:
        """
        Create a chart showing training stability (variance) across runs
        for each DQN model variant.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot create stability chart.")
            return ""
        
        print("Creating training stability comparison chart...")
        
        # If no specific training data, create synthetic example
        if training_data is None:
            training_data = self._generate_stability_example_data()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Chart 1: Win Rate Variance by Model
        models = ['DQNSmall', 'DQNMedium', 'DQNLarge']
        
        if training_data:
            variances = []
            std_devs = []
            mean_performances = []
            
            for model in models:
                if model in training_data:
                    model_data = training_data[model]
                    
                    # Calculate variance across training runs
                    if 'final_win_rates' in model_data:
                        final_wrs = model_data['final_win_rates']
                        variances.append(np.var(final_wrs))
                        std_devs.append(np.std(final_wrs))
                        mean_performances.append(np.mean(final_wrs))
                    else:
                        variances.append(0.0)
                        std_devs.append(0.0)
                        mean_performances.append(0.5)
                else:
                    # Default values if no data
                    variances.append(0.0)
                    std_devs.append(0.0)
                    mean_performances.append(0.5)
            
            # Bar plot of variances
            colors = [self.colors.get(model, f'C{i}') for i, model in enumerate(models)]
            bars = ax1.bar(models, variances, color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels on bars
            for bar, var in zip(bars, variances):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{var:.4f}', ha='center', va='bottom')
            
            ax1.set_title('Training Stability: Win Rate Variance Across Runs', fontsize=14)
            ax1.set_ylabel('Variance in Final Win Rates', fontsize=12)
            ax1.set_xlabel('DQN Model Variant', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Chart 2: Performance vs Stability Scatter
            ax2.scatter(std_devs, mean_performances, s=200, c=colors, alpha=0.7, edgecolors='black')
            
            # Add labels for each point
            for i, model in enumerate(models):
                ax2.annotate(model, (std_devs[i], mean_performances[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax2.set_xlabel('Training Stability (Std Dev)', fontsize=12)
            ax2.set_ylabel('Mean Performance (Win Rate)', fontsize=12)
            ax2.set_title('Performance vs Training Stability', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Add ideal quadrant indicator
            ax2.axhline(y=np.mean(mean_performances), color='red', linestyle='--', alpha=0.5)
            ax2.axvline(x=np.mean(std_devs), color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stability_file = os.path.join(self.output_dir, f"training_stability_comparison_{timestamp}.png")
        plt.savefig(stability_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training stability chart saved to: {stability_file}")
        return stability_file
    
    def _generate_stability_example_data(self) -> Dict:
        """Generate example training stability data for demonstration."""
        np.random.seed(42)  # For reproducible examples
        
        stability_data = {}
        
        # Generate synthetic training data for each model
        models = {
            'DQNSmall': {'base_performance': 0.65, 'stability': 0.08},
            'DQNMedium': {'base_performance': 0.72, 'stability': 0.05}, 
            'DQNLarge': {'base_performance': 0.78, 'stability': 0.04}
        }
        
        for model, params in models.items():
            # Simulate 10 training runs
            final_win_rates = np.random.normal(
                params['base_performance'], 
                params['stability'], 
                10
            )
            final_win_rates = np.clip(final_win_rates, 0.0, 1.0)  # Ensure valid range
            
            stability_data[model] = {
                'final_win_rates': final_win_rates.tolist(),
                'training_curves': [],  # Could add training curves data here
                'mean_performance': float(np.mean(final_win_rates)),
                'std_dev': float(np.std(final_win_rates)),
                'variance': float(np.var(final_win_rates))
            }
        
        return stability_data
    
    def create_comprehensive_dashboard(self, sensitivity_results: Dict = None,
                                    statistical_results: Dict = None,
                                    training_stability: Dict = None) -> str:
        """
        Create a comprehensive analysis dashboard with multiple visualizations.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot create dashboard.")
            return ""
        
        print("Creating comprehensive analysis dashboard...")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Define subplot layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Parameter sensitivity summary (top row)
        if sensitivity_results:
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_sensitivity_summary(ax1, sensitivity_results)
        
        # 2. Performance heatmap (middle left)
        if statistical_results:
            ax2 = fig.add_subplot(gs[1, 0])
            matrix = self._create_win_rate_matrix(statistical_results)
            self._plot_mini_heatmap(ax2, matrix)
        
        # 3. Statistical distributions (middle center)
        if statistical_results:
            ax3 = fig.add_subplot(gs[1, 1])
            self._plot_performance_distributions(ax3, statistical_results)
        
        # 4. Training stability (middle right)
        if training_stability:
            ax4 = fig.add_subplot(gs[1, 2])
            self._plot_stability_summary(ax4, training_stability)
        
        # 5. Agent rankings (bottom left)
        if statistical_results:
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_agent_rankings(ax5, statistical_results)
        
        # 6. Confidence intervals comparison (bottom center)
        if statistical_results:
            ax6 = fig.add_subplot(gs[2, 1])
            self._plot_confidence_intervals(ax6, statistical_results)
        
        # 7. Summary statistics table (bottom right)
        if statistical_results:
            ax7 = fig.add_subplot(gs[2, 2])
            self._plot_summary_table(ax7, statistical_results)
        
        plt.suptitle('Oware AI Comprehensive Analysis Dashboard', fontsize=20, y=0.98)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_file = os.path.join(self.output_dir, f"comprehensive_dashboard_{timestamp}.png")
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive dashboard saved to: {dashboard_file}")
        return dashboard_file
    
    def _plot_sensitivity_summary(self, ax, sensitivity_results):
        """Plot parameter sensitivity summary."""
        # Extract sensitivity coefficients for win_rate metric
        sensitivities = {}
        
        for param, agents in sensitivity_results.items():
            for agent, results in agents.items():
                if isinstance(results, dict) and 'sensitivity_coefficient' in str(results):
                    # This would need to be adapted based on actual data structure
                    pass
        
        # For now, create a placeholder
        ax.text(0.5, 0.5, 'Parameter Sensitivity Summary\n(Coefficients by Agent)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Parameter Sensitivity Overview')
    
    def _plot_mini_heatmap(self, ax, matrix):
        """Plot a mini version of the performance heatmap."""
        if matrix is not None and not matrix.empty:
            sns.heatmap(matrix, ax=ax, cmap='RdYlBu_r', center=0.5, 
                       square=True, cbar=False, annot=False)
        ax.set_title('Win Rate Matrix')
    
    def _plot_performance_distributions(self, ax, statistical_results):
        """Plot performance distributions for agents."""
        agents = []
        win_rates = []
        
        for agent, stats in statistical_results['statistical_summary'].items():
            if 'win_rate' in stats:
                agents.append(agent[:8])  # Truncate names
                win_rates.append(stats['win_rate']['mean'])
        
        if agents:
            ax.bar(range(len(agents)), win_rates, color='skyblue', alpha=0.7)
            ax.set_xticks(range(len(agents)))
            ax.set_xticklabels(agents, rotation=45, ha='right')
            ax.set_ylabel('Win Rate')
            ax.set_title('Performance Distribution')
    
    def _plot_stability_summary(self, ax, training_stability):
        """Plot training stability summary."""
        models = list(training_stability.keys())
        stabilities = [training_stability[m].get('std_dev', 0) for m in models]
        
        ax.bar(models, stabilities, color='lightcoral', alpha=0.7)
        ax.set_ylabel('Std Dev')
        ax.set_title('Training Stability')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_agent_rankings(self, ax, statistical_results):
        """Plot agent performance rankings."""
        rankings = []
        for agent, stats in statistical_results['statistical_summary'].items():
            if 'win_rate' in stats:
                rankings.append((agent[:8], stats['win_rate']['mean']))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        agents, scores = zip(*rankings) if rankings else ([], [])
        y_pos = range(len(agents))
        
        ax.barh(y_pos, scores, color='lightgreen', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(agents)
        ax.set_xlabel('Win Rate')
        ax.set_title('Agent Rankings')
    
    def _plot_confidence_intervals(self, ax, statistical_results):
        """Plot confidence intervals comparison."""
        agents = []
        means = []
        ci_lowers = []
        ci_uppers = []
        
        for agent, stats in statistical_results['statistical_summary'].items():
            if 'win_rate' in stats:
                wr_stats = stats['win_rate']
                agents.append(agent[:6])
                means.append(wr_stats['mean'])
                ci_lowers.append(wr_stats['ci_95_lower'])
                ci_uppers.append(wr_stats['ci_95_upper'])
        
        if agents:
            x_pos = range(len(agents))
            ax.errorbar(x_pos, means, 
                       yerr=[np.array(means) - np.array(ci_lowers), 
                             np.array(ci_uppers) - np.array(means)],
                       fmt='o', capsize=5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(agents, rotation=45)
            ax.set_ylabel('Win Rate')
            ax.set_title('95% Confidence Intervals')
    
    def _plot_summary_table(self, ax, statistical_results):
        """Plot summary statistics table."""
        ax.axis('off')
        
        # Create table data
        table_data = []
        for agent, stats in list(statistical_results['statistical_summary'].items())[:5]:
            if 'win_rate' in stats:
                wr_stats = stats['win_rate']
                row = [
                    agent[:8],
                    f"{wr_stats['mean']:.3f}",
                    f"{wr_stats['std']:.3f}",
                    f"{wr_stats['sample_size']}"
                ]
                table_data.append(row)
        
        if table_data:
            table = ax.table(cellText=table_data,
                           colLabels=['Agent', 'Mean', 'Std', 'N'],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
        
        ax.set_title('Summary Statistics')


def create_all_visualizations(sensitivity_results: Dict = None,
                            statistical_results: Dict = None, 
                            training_stability: Dict = None) -> List[str]:
    """
    Create all available visualizations and return list of file paths.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create visualizations.")
        return []
    
    viz_suite = AdvancedVisualizationSuite()
    created_files = []
    
    try:
        if sensitivity_results:
            sensitivity_plot = viz_suite.create_parameter_sensitivity_plots(sensitivity_results)
            if sensitivity_plot:
                created_files.append(sensitivity_plot)
        
        if statistical_results:
            heatmap_plot = viz_suite.create_performance_heatmap(statistical_results=statistical_results)
            if heatmap_plot:
                created_files.append(heatmap_plot)
        
        if training_stability:
            stability_plot = viz_suite.create_stability_comparison_chart(training_stability)
            if stability_plot:
                created_files.append(stability_plot)
        
        # Create comprehensive dashboard
        dashboard = viz_suite.create_comprehensive_dashboard(
            sensitivity_results, statistical_results, training_stability
        )
        if dashboard:
            created_files.append(dashboard)
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    return created_files


if __name__ == "__main__":
    print("Advanced Visualization Suite")
    print("Import this module and use create_all_visualizations() function")
    print("or create AdvancedVisualizationSuite instance for specific plots.")