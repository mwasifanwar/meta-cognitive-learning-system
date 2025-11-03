import torch
import torch.nn as nn
from typing import Dict, List, Any

class LearningMonitor(nn.Module):
    def __init__(self, threshold_stability: float = 0.1, threshold_confidence: float = 0.7):
        super().__init__()
        self.threshold_stability = threshold_stability
        self.threshold_confidence = threshold_confidence
        
        self.adaptation_policy = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6)
        )
        
        self.performance_history = []
        self.adaptation_history = []
    
    def forward(self, meta_feedback: torch.Tensor) -> torch.Tensor:
        adaptation_scores = self.adaptation_policy(meta_feedback)
        return adaptation_scores
    
    def should_adjust_learning(self, meta_feedback: Dict[str, float]) -> bool:
        stability = meta_feedback.get('learning_stability', 1.0)
        confidence = meta_feedback.get('confidence_level', 0.0)
        adaptation_needed = meta_feedback.get('adaptation_needed', 0.0)
        
        if stability < self.threshold_stability:
            return True
        if confidence < self.threshold_confidence and adaptation_needed > 0.5:
            return True
        
        return adaptation_needed > 0.7
    
    def get_learning_adjustments(self, meta_feedback: Dict[str, float]) -> Dict[str, Any]:
        feedback_tensor = torch.FloatTensor([
            meta_feedback['learning_stability'],
            meta_feedback['convergence_speed'],
            meta_feedback['knowledge_gap'],
            meta_feedback['attention_requirement'],
            meta_feedback['complexity_level'],
            meta_feedback['adaptation_needed'],
            meta_feedback['confidence_level'],
            meta_feedback['efficiency_score']
        ])
        
        with torch.no_grad():
            adaptation_scores = self.forward(feedback_tensor)
            scores_np = adaptation_scores.cpu().numpy()
        
        adjustments = {
            'learning_rate': max(0.0001, min(0.01, 0.001 * (1.0 + scores_np[0]))),
            'batch_size_adjustment': int(scores_np[1] * 32),
            'attention_focus': float(scores_np[2]),
            'complexity_reduction': float(scores_np[3]),
            'knowledge_review': scores_np[4] > 0.5,
            'architecture_adjustment': {
                'layer_modification': scores_np[5] > 0.7
            }
        }
        
        self.adaptation_history.append({
            'meta_feedback': meta_feedback,
            'adjustments': adjustments,
            'timestamp': len(self.performance_history)
        })
        
        return adjustments
    
    def update_performance(self, performance_metrics: Dict[str, float]):
        self.performance_history.append(performance_metrics)
    
    def get_adaptation_effectiveness(self) -> float:
        if len(self.performance_history) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(self.performance_history)):
            if i < len(self.adaptation_history):
                prev_acc = self.performance_history[i-1].get('accuracy', 0.0)
                curr_acc = self.performance_history[i].get('accuracy', 0.0)
                improvements.append(curr_acc - prev_acc)
        
        return np.mean(improvements) if improvements else 0.0