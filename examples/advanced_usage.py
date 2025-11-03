import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import MetaCognitiveEngine
from models import AdaptiveNeuralNetwork

def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
    
    return train_loader, test_loader

class MNISTModel(AdaptiveNeuralNetwork):
    def __init__(self):
        super().__init__(input_size=784, hidden_sizes=[128, 64], output_size=10, adaptive_layers=True)
    
    def forward(self, x):
        x = x.view(-1, 784)
        return super().forward(x)

def advanced_demo():
    print("=== Advanced Meta-Cognitive Learning Demo ===")
    print("MNIST Classification with Self-Improving AI")
    print("Created by mwasifanwar")
    
    train_loader, test_loader = load_mnist_data()
    base_model = MNISTModel()
    
    meta_engine = MetaCognitiveEngine(base_model, learning_rate=0.001)
    
    task_description = "Handwritten digit recognition using MNIST dataset with adaptive learning"
    
    print("Starting advanced meta-cognitive learning...")
    meta_engine.learn(train_loader, task_description, num_epochs=3)
    
    print("\nAnalyzing learning patterns and generating insights...")
    insights = meta_engine.get_learning_insights()
    
    print("Meta-Cognitive Insights:")
    for key, value in insights.items():
        if key != 'knowledge_base_summary':
            print(f"  {key}: {value}")
    
    print("\nDetailed Knowledge Base Analysis:")
    kb_summary = insights['knowledge_base_summary']
    for key, value in kb_summary.items():
        print(f"  {key}: {value}")
    
    print("\nLearning History Analysis:")
    print(f"Total learning steps: {len(meta_engine.learning_history)}")
    print(f"Adaptation events: {len([h for h in meta_engine.learning_history if h.get('learning_adjustments')])}")
    
    return meta_engine

if __name__ == "__main__":
    engine = advanced_demo()