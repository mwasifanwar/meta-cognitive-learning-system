import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import MetaCognitiveEngine
from models import AdaptiveNeuralNetwork

def create_sample_data():
    torch.manual_seed(42)
    np.random.seed(42)
    
    X = torch.randn(1000, 20)
    y = torch.randint(0, 3, (1000,))
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def demo_meta_cognitive_learning():
    print("=== Meta-Cognitive Learning System Demo ===")
    print("Created by mwasifanwar")
    
    base_model = AdaptiveNeuralNetwork(
        input_size=20,
        hidden_sizes=[64, 32, 16],
        output_size=3,
        adaptive_layers=True
    )
    
    meta_engine = MetaCognitiveEngine(base_model, learning_rate=0.01)
    
    data_loader = create_sample_data()
    task_description = "Multi-class classification with 20 features and 3 classes"
    
    print("Starting meta-cognitive learning process...")
    meta_engine.learn(data_loader, task_description, num_epochs=5)
    
    print("\nLearning completed! Generating insights...")
    insights = meta_engine.get_learning_insights()
    
    print(f"Performance Trend: {insights['performance_trend']:.4f}")
    print(f"Learning Efficiency: {insights['learning_efficiency']:.4f}")
    print(f"Optimal Conditions: {insights['optimal_learning_conditions']}")
    
    print("\nKnowledge Base Summary:")
    for key, value in insights['knowledge_base_summary'].items():
        print(f"  {key}: {value}")
    
    return meta_engine

if __name__ == "__main__":
    engine = demo_meta_cognitive_learning()