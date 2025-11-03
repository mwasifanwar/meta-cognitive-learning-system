import logging
import json
import pickle
import numpy as np
import torch
from datetime import datetime

def setup_logging(name="meta_cognitive_system"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{name}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def save_results(results, filename="meta_cognitive_results.json"):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def load_results(filename="meta_cognitive_results.json"):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def calculate_learning_metrics(history):
    if not history:
        return {}
    
    losses = [entry['loss'] for entry in history]
    accuracies = [entry['accuracy'] for entry in history]
    
    return {
        'final_accuracy': accuracies[-1],
        'best_accuracy': max(accuracies),
        'average_loss': np.mean(losses),
        'convergence_speed': len(history),
        'stability': np.std(accuracies)
    }

def create_performance_report(engine, task_description):
    insights = engine.get_learning_insights()
    metrics = calculate_learning_metrics(engine.learning_history)
    
    report = {
        'task_description': task_description,
        'performance_metrics': metrics,
        'meta_cognitive_insights': insights,
        'learning_efficiency': engine._calculate_learning_efficiency(),
        'adaptation_effectiveness': engine.learning_monitor.get_adaptation_effectiveness(),
        'knowledge_base_summary': engine.knowledge_base.get_summary(),
        'timestamp': datetime.now().isoformat()
    }
    
    return report