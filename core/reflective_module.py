import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any

class ReflectiveModule(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.learning_state_analyzer = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.meta_feedback_generator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        
        self.pattern_recognizer = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
    def forward(self, learning_state: torch.Tensor) -> torch.Tensor:
        encoded_state = self.learning_state_analyzer(learning_state)
        meta_feedback = self.meta_feedback_generator(encoded_state)
        return meta_feedback
    
    def analyze_learning_state(self, learning_state: Dict[str, Any]) -> Dict[str, float]:
        state_tensor = self._state_to_tensor(learning_state)
        
        with torch.no_grad():
            meta_feedback = self.forward(state_tensor)
            meta_feedback_np = meta_feedback.cpu().numpy()
        
        feedback_dict = {
            'learning_stability': float(meta_feedback_np[0]),
            'convergence_speed': float(meta_feedback_np[1]),
            'knowledge_gap': float(meta_feedback_np[2]),
            'attention_requirement': float(meta_feedback_np[3]),
            'complexity_level': float(meta_feedback_np[4]),
            'adaptation_needed': float(meta_feedback_np[5]),
            'confidence_level': float(meta_feedback_np[6]),
            'efficiency_score': float(meta_feedback_np[7])
        }
        
        return feedback_dict
    
    def _state_to_tensor(self, learning_state: Dict[str, Any]) -> torch.Tensor:
        features = [
            learning_state.get('gradient_norm', 0.0),
            np.mean(learning_state.get('recent_losses', [0.0])),
            np.mean(learning_state.get('recent_accuracies', [0.0])),
            learning_state.get('model_confidence', 0.0)
        ]
        return torch.FloatTensor(features)
    
    def extract_learning_patterns(self, learning_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(learning_history) < 5:
            return {}
        
        losses = [entry['loss'] for entry in learning_history]
        accuracies = [entry['accuracy'] for entry in learning_history]
        
        loss_trend = self._calculate_trend(losses)
        accuracy_trend = self._calculate_trend(accuracies)
        
        learning_oscillations = self._detect_oscillations(accuracies)
        convergence_points = self._find_convergence_points(losses)
        
        return {
            'loss_trend': loss_trend,
            'accuracy_trend': accuracy_trend,
            'oscillation_frequency': learning_oscillations,
            'convergence_epochs': convergence_points,
            'learning_consistency': self._calculate_consistency(accuracies)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)
    
    def _detect_oscillations(self, values: List[float]) -> int:
        if len(values) < 3:
            return 0
        
        oscillations = 0
        for i in range(1, len(values)-1):
            if (values[i] > values[i-1] and values[i] > values[i+1]) or \
               (values[i] < values[i-1] and values[i] < values[i+1]):
                oscillations += 1
        
        return oscillations
    
    def _find_convergence_points(self, losses: List[float]) -> List[int]:
        if len(losses) < 10:
            return []
        
        convergence_points = []
        window_size = 5
        convergence_threshold = 0.01
        
        for i in range(window_size, len(losses)):
            window = losses[i-window_size:i]
            if np.std(window) < convergence_threshold:
                convergence_points.append(i)
        
        return convergence_points
    
    def _calculate_consistency(self, values: List[float]) -> float:
        if len(values) < 2:
            return 1.0
        return float(1.0 - (np.std(values) / (np.mean(values) + 1e-8)))