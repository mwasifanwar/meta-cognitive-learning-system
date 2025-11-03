class Config:
    META_LEARNING_RATE = 0.001
    BASE_LEARNING_RATE = 0.01
    REFLECTION_INTERVAL = 100
    KNOWLEDGE_UPDATE_FREQUENCY = 10
    PERFORMANCE_THRESHOLD = 0.75
    ADAPTATION_CONFIDENCE = 0.8
    
    MODEL_CONFIG = {
        'hidden_sizes': [128, 64, 32],
        'adaptive_layers': True,
        'attention_heads': 4
    }
    
    MEMORY_CONFIG = {
        'episodic_capacity': 1000,
        'semantic_relationships': True
    }
    
    @classmethod
    def to_dict(cls):
        return {key: value for key, value in cls.__dict__.items() 
                if not key.startswith('_') and not callable(value)}