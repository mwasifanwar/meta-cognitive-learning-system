import argparse
from examples.demo_meta_cognition import demo_meta_cognitive_learning
from examples.advanced_usage import advanced_demo

def main():
    parser = argparse.ArgumentParser(description="Meta-Cognitive Learning System")
    parser.add_argument('--mode', type=str, choices=['demo', 'advanced', 'train'], default='demo')
    parser.add_argument('--task', type=str, help='Task description for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        print("Running Meta-Cognitive Learning Demo")
        demo_meta_cognitive_learning()
    
    elif args.mode == 'advanced':
        print("Running Advanced Meta-Cognitive Demo")
        advanced_demo()
    
    elif args.mode == 'train':
        print(f"Training mode with task: {args.task}")
        print(f"Epochs: {args.epochs}")

if __name__ == "__main__":
    main()