import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, adaptive_layers: bool = True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.adaptive_layers = adaptive_layers
        
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(prev_size, hidden_size))
            if adaptive_layers:
                self.layers.append(AdaptiveActivation(hidden_size))
            else:
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        self.output_layer = nn.Linear(prev_size, output_size)
        self.attention_mechanism = SelfAttention(prev_size) if adaptive_layers else None
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        if self.attention_mechanism:
            x = self.attention_mechanism(x)
        
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)
    
    def adapt_complexity(self, complexity_level: float):
        if not self.adaptive_layers:
            return
        
        for layer in self.layers:
            if isinstance(layer, AdaptiveActivation):
                layer.set_adaptation_level(complexity_level)

class AdaptiveActivation(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.adaptation_level = 1.0
    
    def forward(self, x):
        base_activation = F.leaky_relu(x, negative_slope=0.01)
        adapted_activation = self.alpha * torch.tanh(self.beta * x)
        return self.adaptation_level * adapted_activation + (1 - self.adaptation_level) * base_activation
    
    def set_adaptation_level(self, level: float):
        self.adaptation_level = max(0.0, min(1.0, level))

class SelfAttention(nn.Module):
    def __init__(self, size: int, heads: int = 4):
        super().__init__()
        self.size = size
        self.heads = heads
        self.head_dim = size // heads
        
        self.query = nn.Linear(size, size)
        self.key = nn.Linear(size, size)
        self.value = nn.Linear(size, size)
        self.out = nn.Linear(size, size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.query(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        attention = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        out = torch.matmul(attention, V).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.out(out)

class MetaLearningModel(nn.Module):
    def __init__(self, base_model: nn.Module, meta_hidden_size: int = 64):
        super().__init__()
        self.base_model = base_model
        self.meta_learner = nn.Sequential(
            nn.Linear(self._get_base_parameter_count(), meta_hidden_size),
            nn.ReLU(),
            nn.Linear(meta_hidden_size, meta_hidden_size),
            nn.ReLU(),
            nn.Linear(meta_hidden_size, self._get_base_parameter_count())
        )
    
    def _get_base_parameter_count(self) -> int:
        return sum(p.numel() for p in self.base_model.parameters())
    
    def forward(self, x):
        return self.base_model(x)
    
    def meta_update(self, gradients: List[torch.Tensor]):
        flat_gradients = torch.cat([g.flatten() for g in gradients])
        meta_update = self.meta_learner(flat_gradients)
        
        return meta_update

class SelfImprovingModel(nn.Module):
    def __init__(self, base_architecture: nn.Module, improvement_strategy: str = "gradient_based"):
        super().__init__()
        self.base_model = base_architecture
        self.improvement_strategy = improvement_strategy
        self.performance_tracker = PerformanceTracker()
        self.architecture_optimizer = ArchitectureOptimizer(base_architecture)
    
    def forward(self, x):
        return self.base_model(x)
    
    def self_improve(self, validation_loader):
        current_performance = self.performance_tracker.evaluate(self.base_model, validation_loader)
        improvement_plan = self.architecture_optimizer.generate_improvement_plan(current_performance)
        
        if improvement_plan['should_modify']:
            self.base_model = self.architecture_optimizer.apply_improvements(
                self.base_model, improvement_plan
            )
        
        return improvement_plan