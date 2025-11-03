import torch
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ReflectiveModule

def test_reflective_module_forward():
    module = ReflectiveModule()
    test_input = torch.FloatTensor([0.5, 0.3, 0.8, 0.2])
    
    output = module(test_input)
    
    assert output.shape == torch.Size([8])
    assert torch.all(output >= -1) and torch.all(output <= 1)

def test_learning_pattern_extraction():
    module = ReflectiveModule()
    
    learning_history = [
        {'loss': 0.8, 'accuracy': 0.5},
        {'loss': 0.7, 'accuracy': 0.6},
        {'loss': 0.6, 'accuracy': 0.7},
        {'loss': 0.5, 'accuracy': 0.75},
        {'loss': 0.5, 'accuracy': 0.76}
    ]
    
    patterns = module.extract_learning_patterns(learning_history)
    
    assert 'loss_trend' in patterns
    assert 'accuracy_trend' in patterns
    assert 'oscillation_frequency' in patterns
    assert 'convergence_epochs' in patterns