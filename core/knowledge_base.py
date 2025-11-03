import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

class KnowledgeBase:
    def __init__(self, storage_path: str = "knowledge_base.pkl"):
        self.storage_path = storage_path
        self.task_knowledge = {}
        self.learning_strategies = {}
        self.performance_patterns = {}
        self.meta_cognitive_rules = {}
        
        self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self):
        self.meta_cognitive_rules = {
            'high_loss_oscillation': 'reduce_learning_rate',
            'plateaued_accuracy': 'increase_model_complexity',
            'overfitting_detected': 'add_regularization',
            'slow_convergence': 'adjust_optimizer',
            'high_confidence_low_accuracy': 'review_fundamentals'
        }
        
        self.learning_strategies = {
            'progressive_learning': {
                'description': 'Gradually increase complexity',
                'success_rate': 0.85,
                'applicable_tasks': ['classification', 'regression']
            },
            'adaptive_curriculum': {
                'description': 'Dynamic difficulty adjustment',
                'success_rate': 0.78,
                'applicable_tasks': ['reinforcement_learning', 'sequential_tasks']
            }
        }
    
    def store_task_context(self, task_description: str):
        task_id = hash(task_description)
        self.task_knowledge[task_id] = {
            'description': task_description,
            'created_at': datetime.now().isoformat(),
            'learning_sessions': [],
            'best_strategies': [],
            'challenges_faced': []
        }
        return task_id
    
    def update_knowledge(self, learning_session: Dict[str, Any]):
        task_id = list(self.task_knowledge.keys())[-1] if self.task_knowledge else None
        if task_id:
            self.task_knowledge[task_id]['learning_sessions'].append(learning_session)
            
            performance = learning_session.get('performance', {})
            if performance.get('accuracy', 0) > 0.8:
                strategies = learning_session.get('successful_strategies', [])
                if strategies:
                    self.task_knowledge[task_id]['best_strategies'].extend(strategies)
            
            failed_strategies = learning_session.get('failed_strategies', [])
            if failed_strategies:
                self.task_knowledge[task_id]['challenges_faced'].extend(failed_strategies)
    
    def retrieve_relevant_knowledge(self, task_description: str) -> Dict[str, Any]:
        task_keywords = set(task_description.lower().split())
        relevant_knowledge = {}
        
        for task_id, knowledge in self.task_knowledge.items():
            stored_description = knowledge['description'].lower()
            stored_keywords = set(stored_description.split())
            
            similarity = len(task_keywords.intersection(stored_keywords)) / len(task_keywords.union(stored_keywords))
            if similarity > 0.3:
                relevant_knowledge[task_id] = {
                    'similarity': similarity,
                    'knowledge': knowledge,
                    'best_strategies': knowledge.get('best_strategies', [])[:3]
                }
        
        return relevant_knowledge
    
    def extract_learning_insights(self) -> Dict[str, Any]:
        total_sessions = sum(len(task['learning_sessions']) for task in self.task_knowledge.values())
        successful_strategies = []
        
        for task_id, knowledge in self.task_knowledge.items():
            successful_strategies.extend(knowledge.get('best_strategies', []))
        
        if successful_strategies:
            best_strategy = max(successful_strategies, key=lambda x: x.get('improvement', 0))
        else:
            best_strategy = {}
        
        return {
            'total_learning_sessions': total_sessions,
            'tasks_learned': len(self.task_knowledge),
            'most_effective_strategy': best_strategy,
            'average_improvement': np.mean([s.get('improvement', 0) for s in successful_strategies]) if successful_strategies else 0,
            'common_challenges': self._extract_common_challenges()
        }
    
    def _extract_common_challenges(self) -> List[str]:
        all_challenges = []
        for task_knowledge in self.task_knowledge.values():
            all_challenges.extend([strategy.get('adjustments', {}) for strategy in task_knowledge.get('challenges_faced', [])])
        
        challenge_counts = {}
        for challenge in all_challenges:
            challenge_key = str(challenge)
            challenge_counts[challenge_key] = challenge_counts.get(challenge_key, 0) + 1
        
        return sorted(challenge_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def get_summary(self) -> Dict[str, Any]:
        insights = self.extract_learning_insights()
        return {
            'knowledge_base_size': len(self.task_knowledge),
            'total_insights': insights['total_learning_sessions'],
            'effectiveness_score': insights['average_improvement'],
            'learning_efficiency': self._calculate_learning_efficiency(),
            'adaptation_intelligence': self._calculate_adaptation_intelligence()
        }
    
    def _calculate_learning_efficiency(self) -> float:
        if not self.task_knowledge:
            return 0.0
        
        efficiency_scores = []
        for task_knowledge in self.task_knowledge.values():
            sessions = task_knowledge.get('learning_sessions', [])
            if len(sessions) >= 2:
                first_acc = sessions[0].get('performance', {}).get('accuracy', 0)
                last_acc = sessions[-1].get('performance', {}).get('accuracy', 0)
                improvement = last_acc - first_acc
                efficiency = improvement / len(sessions) if len(sessions) > 0 else 0
                efficiency_scores.append(efficiency)
        
        return np.mean(efficiency_scores) if efficiency_scores else 0.0
    
    def _calculate_adaptation_intelligence(self) -> float:
        successful_adaptations = 0
        total_adaptations = 0
        
        for task_knowledge in self.task_knowledge.values():
            best_strategies = task_knowledge.get('best_strategies', [])
            challenges = task_knowledge.get('challenges_faced', [])
            
            successful_adaptations += len(best_strategies)
            total_adaptations += len(best_strategies) + len(challenges)
        
        return successful_adaptations / total_adaptations if total_adaptations > 0 else 0.0
    
    def save_knowledge(self):
        with open(self.storage_path, 'wb') as f:
            pickle.dump({
                'task_knowledge': self.task_knowledge,
                'learning_strategies': self.learning_strategies,
                'performance_patterns': self.performance_patterns,
                'meta_cognitive_rules': self.meta_cognitive_rules
            }, f)
    
    def load_knowledge(self):
        try:
            with open(self.storage_path, 'rb') as f:
                data = pickle.load(f)
                self.task_knowledge = data.get('task_knowledge', {})
                self.learning_strategies = data.get('learning_strategies', {})
                self.performance_patterns = data.get('performance_patterns', {})
                self.meta_cognitive_rules = data.get('meta_cognitive_rules', {})
        except FileNotFoundError:
            pass