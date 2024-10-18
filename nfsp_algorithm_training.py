import argparse
from nfsp_algorithm import NFSPChess
import torch

def main():
    parser = argparse.ArgumentParser(description='Train NFSP Chess models')
    parser.add_argument('--pgn_path', type=str, required=True, help='Path to PGN file')
    parser.add_argument('--model_type', choices=['standard', 'puzzle'], required=True, 
                        help='Type of training data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save model')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Initialize NFSP
    nfsp = NFSPChess(batch_size=args.batch_size)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        nfsp.load_model(args.resume)
    
    # Train model
    print(f"Training NFSP model on {args.model_type} data...")
    print(f"Using device: {nfsp.device}")
    
    try:
        losses = nfsp.train_on_pgn(args.pgn_path, num_epochs=args.epochs)
        
        # Save final model
        nfsp.save_model(args.save_path)
        print(f"Model saved to {args.save_path}")
        
        # Plot training progress
        nfsp.plot_training_progress()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        nfsp.save_model(f"{args.save_path}_interrupted.pt")
        print(f"Checkpoint saved to {args.save_path}_interrupted.pt")

if __name__ == "__main__":
    main()