import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional

class MetaCognitiveEngine:
    def __init__(self, base_model: nn.Module, learning_rate: float = 0.001):
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model.to(self.device)
        
        self.reflective_module = ReflectiveModule()
        self.learning_monitor = LearningMonitor()
        self.knowledge_base = KnowledgeBase()
        
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=learning_rate)
        self.meta_optimizer = optim.Adam(list(self.reflective_module.parameters()) + 
                                       list(self.learning_monitor.parameters()), lr=learning_rate)
        
        self.learning_history = []
        self.performance_metrics = {}
        
    def learn(self, data_loader, task_description: str, num_epochs: int = 10):
        self.knowledge_base.store_task_context(task_description)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            for batch_idx, (data, targets) in enumerate(data_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                current_state = self._get_current_learning_state()
                meta_feedback = self.reflective_module.analyze_learning_state(current_state)
                
                if self.learning_monitor.should_adjust_learning(meta_feedback):
                    learning_adjustments = self.learning_monitor.get_learning_adjustments(meta_feedback)
                    self._apply_learning_adjustments(learning_adjustments)
                
                self.optimizer.zero_grad()
                outputs = self.base_model(data)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()
                
                learning_snapshot = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss.item(),
                    'accuracy': (predicted == targets).float().mean().item(),
                    'meta_feedback': meta_feedback,
                    'learning_adjustments': learning_adjustments if self.learning_monitor.should_adjust_learning(meta_feedback) else None
                }
                self.learning_history.append(learning_snapshot)
            
            epoch_accuracy = correct_predictions / total_samples
            self.performance_metrics[epoch] = {
                'loss': epoch_loss / len(data_loader),
                'accuracy': epoch_accuracy
            }
            
            self._update_meta_cognitive_knowledge(epoch)
    
    def _get_current_learning_state(self) -> Dict[str, Any]:
        current_gradients = []
        for param in self.base_model.parameters():
            if param.grad is not None:
                current_gradients.append(param.grad.flatten())
        
        grad_norm = torch.cat(current_gradients).norm().item() if current_gradients else 0.0
        
        return {
            'gradient_norm': grad_norm,
            'recent_losses': [hist['loss'] for hist in self.learning_history[-10:]],
            'recent_accuracies': [hist['accuracy'] for hist in self.learning_history[-10:]],
            'model_confidence': self._calculate_model_confidence()
        }
    
    def _calculate_model_confidence(self) -> float:
        if not self.learning_history:
            return 0.0
        recent_accuracies = [hist['accuracy'] for hist in self.learning_history[-5:]]
        return np.mean(recent_accuracies) if recent_accuracies else 0.0
    
    def _apply_learning_adjustments(self, adjustments: Dict[str, Any]):
        if 'learning_rate' in adjustments:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = adjustments['learning_rate']
        
        if 'architecture_adjustment' in adjustments:
            self._adjust_model_architecture(adjustments['architecture_adjustment'])
    
    def _adjust_model_architecture(self, adjustment: Dict[str, Any]):
        pass
    
    def _update_meta_cognitive_knowledge(self, epoch: int):
        recent_performance = self.performance_metrics.get(epoch, {})
        learning_patterns = self.reflective_module.extract_learning_patterns(self.learning_history)
        
        self.knowledge_base.update_knowledge({
            'epoch': epoch,
            'performance': recent_performance,
            'learning_patterns': learning_patterns,
            'successful_strategies': self._extract_successful_strategies(),
            'failed_strategies': self._extract_failed_strategies()
        })
    
    def _extract_successful_strategies(self) -> List[Dict[str, Any]]:
        successful = []
        for i in range(1, len(self.learning_history)):
            prev = self.learning_history[i-1]
            curr = self.learning_history[i]
            
            if curr['accuracy'] > prev['accuracy'] and curr['loss'] < prev['loss']:
                if curr.get('learning_adjustments'):
                    successful.append({
                        'adjustments': curr['learning_adjustments'],
                        'improvement': curr['accuracy'] - prev['accuracy']
                    })
        return successful
    
    def _extract_failed_strategies(self) -> List[Dict[str, Any]]:
        failed = []
        for i in range(1, len(self.learning_history)):
            prev = self.learning_history[i-1]
            curr = self.learning_history[i]
            
            if curr['accuracy'] < prev['accuracy'] and curr['loss'] > prev['loss']:
                if curr.get('learning_adjustments'):
                    failed.append({
                        'adjustments': curr['learning_adjustments'],
                        'deterioration': prev['accuracy'] - curr['accuracy']
                    })
        return failed
    
    def get_learning_insights(self) -> Dict[str, Any]:
        return {
            'performance_trend': self._calculate_performance_trend(),
            'optimal_learning_conditions': self._identify_optimal_conditions(),
            'learning_efficiency': self._calculate_learning_efficiency(),
            'knowledge_base_summary': self.knowledge_base.get_summary()
        }
    
    def _calculate_performance_trend(self) -> float:
        if len(self.performance_metrics) < 2:
            return 0.0
        
        recent_accuracies = [metrics['accuracy'] for metrics in self.performance_metrics.values()]
        return np.polyfit(range(len(recent_accuracies)), recent_accuracies, 1)[0]
    
    def _identify_optimal_conditions(self) -> Dict[str, Any]:
        successful_strategies = self._extract_successful_strategies()
        if not successful_strategies:
            return {}
        
        best_strategy = max(successful_strategies, key=lambda x: x['improvement'])
        return best_strategy['adjustments']
    
    def _calculate_learning_efficiency(self) -> float:
        if not self.learning_history:
            return 0.0
        
        improvements = []
        for i in range(1, len(self.learning_history)):
            if i % 10 == 0:
                prev_acc = self.learning_history[i-10]['accuracy']
                curr_acc = self.learning_history[i]['accuracy']
                improvements.append(curr_acc - prev_acc)
        
        return np.mean(improvements) if improvements else 0.0
