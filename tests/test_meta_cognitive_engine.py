import torch
import torch.nn as nn
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import MetaCognitiveEngine
from models import AdaptiveNeuralNetwork

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 3)
    
    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)

def test_meta_cognitive_engine_initialization():
    base_model = TestModel()
    engine = MetaCognitiveEngine(base_model)
    
    assert engine.base_model == base_model
    assert engine.reflective_module is not None
    assert engine.learning_monitor is not None
    assert engine.knowledge_base is not None

def test_learning_state_analysis():
    base_model = TestModel()
    engine = MetaCognitiveEngine(base_model)
    
    learning_state = engine._get_current_learning_state()
    
    assert 'gradient_norm' in learning_state
    assert 'recent_losses' in learning_state
    assert 'recent_accuracies' in learning_state
    assert 'model_confidence' in learning_state

def test_meta_feedback_generation():
    base_model = AdaptiveNeuralNetwork(10, [16, 8], 3)
    engine = MetaCognitiveEngine(base_model)
    
    learning_state = {
        'gradient_norm': 0.5,
        'recent_losses': [0.8, 0.7, 0.6],
        'recent_accuracies': [0.5, 0.6, 0.7],
        'model_confidence': 0.6
    }
    
    meta_feedback = engine.reflective_module.analyze_learning_state(learning_state)
    
    assert 'learning_stability' in meta_feedback
    assert 'convergence_speed' in meta_feedback
    assert 'knowledge_gap' in meta_feedback
    assert all(0 <= value <= 1 for value in meta_feedback.values())