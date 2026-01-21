"""
Experiment Tracker for Air Quality Forecasting

This module provides tools for systematic experimentation,
result tracking, and performance analysis.

Author: Reine Mizero
Date: January 2026
"""

import json
import csv
import os
from datetime import datetime
import numpy as np

class ExperimentTracker:
    """
    Track and manage machine learning experiments
    """
    
    def __init__(self, experiment_dir="experiments"):
        """
        Initialize experiment tracker
        
        Parameters:
        - experiment_dir: Directory to store experiment results
        """
        self.experiment_dir = experiment_dir
        self.experiments = []
        self.current_experiment = None
        
        # Create experiment directory if it doesn't exist
        os.makedirs(experiment_dir, exist_ok=True)
        
    def start_experiment(self, name, description, parameters):
        """
        Start a new experiment
        
        Parameters:
        - name: Experiment name
        - description: Experiment description
        - parameters: Dictionary of experiment parameters
        """
        experiment = {
            'id': len(self.experiments) + 1,
            'name': name,
            'description': description,
            'parameters': parameters,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'results': {}
        }
        
        self.experiments.append(experiment)
        self.current_experiment = experiment
        
        print(f"🔬 Started Experiment {experiment['id']}: {name}")
        return experiment['id']
    
    def log_metric(self, metric_name, value, step=None):
        """
        Log a metric for the current experiment
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        if 'metrics' not in self.current_experiment:
            self.current_experiment['metrics'] = {}
        
        if metric_name not in self.current_experiment['metrics']:
            self.current_experiment['metrics'][metric_name] = []
        
        metric_entry = {'value': value}
        if step is not None:
            metric_entry['step'] = step
        
        self.current_experiment['metrics'][metric_name].append(metric_entry)
    
    def log_result(self, result_dict):
        """
        Log experiment results
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment['results'].update(result_dict)
    
    def end_experiment(self, status='completed'):
        """
        End the current experiment
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment.")
        
        self.current_experiment['end_time'] = datetime.now().isoformat()
        self.current_experiment['status'] = status
        
        # Calculate duration
        start_time = datetime.fromisoformat(self.current_experiment['start_time'])
        end_time = datetime.fromisoformat(self.current_experiment['end_time'])
        duration = (end_time - start_time).total_seconds()
        self.current_experiment['duration_seconds'] = duration
        
        print(f"✅ Completed Experiment {self.current_experiment['id']} in {duration:.1f}s")
        
        # Save experiment
        self.save_experiment(self.current_experiment)
        self.current_experiment = None
    
    def save_experiment(self, experiment):
        """
        Save experiment to file
        """
        filename = f"experiment_{experiment['id']:03d}_{experiment['name']}.json"
        filepath = os.path.join(self.experiment_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(experiment, f, indent=2, default=str)
    
    def save_all_experiments(self, filename="all_experiments.json"):
        """
        Save all experiments to a single file
        """
        filepath = os.path.join(self.experiment_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
    
    def export_to_csv(self, filename="experiment_results.csv"):
        """
        Export experiment results to CSV
        """
        if not self.experiments:
            print("No experiments to export")
            return
        
        filepath = os.path.join(self.experiment_dir, filename)
        
        # Prepare data for CSV
        csv_data = []
        for exp in self.experiments:
            row = {
                'experiment_id': exp['id'],
                'name': exp['name'],
                'description': exp['description'],
                'status': exp['status'],
                'duration_seconds': exp.get('duration_seconds', 0)
            }
            
            # Add parameters
            if 'parameters' in exp:
                for key, value in exp['parameters'].items():
                    row[f'param_{key}'] = value
            
            # Add results
            if 'results' in exp:
                for key, value in exp['results'].items():
                    row[f'result_{key}'] = value
            
            csv_data.append(row)
        
        # Write CSV
        if csv_data:
            fieldnames = csv_data[0].keys()
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"📊 Exported {len(csv_data)} experiments to {filepath}")
    
    def get_best_experiment(self, metric='val_rmse', minimize=True):
        """
        Get the best experiment based on a metric
        """
        if not self.experiments:
            return None
        
        valid_experiments = [
            exp for exp in self.experiments 
            if exp['status'] == 'completed' and 
               'results' in exp and 
               metric in exp['results']
        ]
        
        if not valid_experiments:
            return None
        
        if minimize:
            return min(valid_experiments, key=lambda x: x['results'][metric])
        else:
            return max(valid_experiments, key=lambda x: x['results'][metric])
    
    def get_experiment_summary(self):
        """
        Get summary of all experiments
        """
        if not self.experiments:
            return "No experiments found"
        
        completed = len([e for e in self.experiments if e['status'] == 'completed'])
        failed = len([e for e in self.experiments if e['status'] == 'failed'])
        running = len([e for e in self.experiments if e['status'] == 'running'])
        
        summary = f"""
Experiment Summary:
- Total experiments: {len(self.experiments)}
- Completed: {completed}
- Failed: {failed}  
- Running: {running}
        """
        
        if completed > 0:
            best_exp = self.get_best_experiment()
            if best_exp:
                summary += f"\nBest experiment: #{best_exp['id']} - {best_exp['name']}"
                summary += f"\nBest RMSE: {best_exp['results'].get('val_rmse', 'N/A')}"
        
        return summary
    
    def compare_experiments(self, metric='val_rmse'):
        """
        Compare experiments by a specific metric
        """
        valid_experiments = [
            exp for exp in self.experiments 
            if exp['status'] == 'completed' and 
               'results' in exp and 
               metric in exp['results']
        ]
        
        if not valid_experiments:
            return "No valid experiments to compare"
        
        # Sort by metric
        sorted_experiments = sorted(valid_experiments, key=lambda x: x['results'][metric])
        
        comparison = f"Experiments ranked by {metric}:\n"
        comparison += "-" * 50 + "\n"
        
        for i, exp in enumerate(sorted_experiments[:10], 1):  # Top 10
            comparison += f"{i:2d}. Exp #{exp['id']:2d}: {exp['results'][metric]:8.2f} - {exp['name']}\n"
        
        return comparison
    
    def load_experiments(self, directory=None):
        """
        Load experiments from directory
        """
        if directory is None:
            directory = self.experiment_dir
        
        loaded_count = 0
        for filename in os.listdir(directory):
            if filename.startswith('experiment_') and filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r') as f:
                        experiment = json.load(f)
                        self.experiments.append(experiment)
                        loaded_count += 1
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        print(f"📁 Loaded {loaded_count} experiments from {directory}")
        return loaded_count

class ExperimentConfig:
    """
    Configuration generator for systematic experiments
    """
    
    @staticmethod
    def generate_lstm_configs():
        """
        Generate LSTM experiment configurations
        """
        configs = []
        
        # Simple LSTM variations
        for units in [32, 64, 128]:
            for dropout in [0.0, 0.1, 0.2]:
                for lr in [0.001, 0.005, 0.0005]:
                    config = {
                        'model_type': 'simple_lstm',
                        'model_params': {'units': units, 'dropout': dropout},
                        'compile_params': {
                            'optimizer': f'adam_lr_{lr}',
                            'learning_rate': lr,
                            'loss': 'mse'
                        },
                        'fit_params': {'epochs': 50, 'batch_size': 32},
                        'description': f'Simple LSTM, units={units}, dropout={dropout}, lr={lr}'
                    }
                    configs.append(config)
        
        return configs
    
    @staticmethod
    def generate_comprehensive_configs():
        """
        Generate comprehensive experiment configurations
        """
        configs = []
        
        # Model architectures
        architectures = [
            ('simple_lstm', {'units': 32, 'dropout': 0.0}),
            ('simple_lstm', {'units': 64, 'dropout': 0.1}),
            ('deep_lstm', {'units': [64, 32], 'dropout': 0.2}),
            ('deep_lstm', {'units': [128, 64, 32], 'dropout': 0.3}),
            ('bidirectional_lstm', {'units': 64, 'dropout': 0.2}),
            ('gru_model', {'units': [64, 32], 'dropout': 0.2}),
        ]
        
        # Hyperparameters
        learning_rates = [0.0005, 0.001, 0.002, 0.005]
        batch_sizes = [16, 32, 64, 128]
        optimizers = ['adam', 'rmsprop']
        
        config_id = 1
        for arch_name, arch_params in architectures:
            for lr in learning_rates:
                for batch_size in batch_sizes:
                    for optimizer in optimizers:
                        if config_id > 20:  # Limit to 20 configurations
                            break
                        
                        config = {
                            'config_id': config_id,
                            'model_type': arch_name,
                            'model_params': arch_params,
                            'compile_params': {
                                'optimizer': optimizer,
                                'learning_rate': lr,
                                'loss': 'mse'
                            },
                            'fit_params': {
                                'epochs': 50, 
                                'batch_size': batch_size
                            },
                            'description': f'{arch_name}, {optimizer}, lr={lr}, batch={batch_size}'
                        }
                        configs.append(config)
                        config_id += 1
        
        return configs

if __name__ == "__main__":
    print("Experiment Tracker for Air Quality Forecasting")
    
    # Example usage
    tracker = ExperimentTracker()
    
    # Generate some example configurations
    configs = ExperimentConfig.generate_comprehensive_configs()
    print(f"Generated {len(configs)} experiment configurations")
    
    print("\nExample experiment configuration:")
    if configs:
        print(json.dumps(configs[0], indent=2))
