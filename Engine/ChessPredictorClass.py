import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess
import json

class ChessFeatureEncoder:
    def __init__(self):
        self.piece_mapping = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
        }
    
    def encode_board(self, board):
        encoded = np.zeros((12, 8, 8), dtype=np.float32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = square // 8
                col = square % 8
                piece_channel = self.piece_mapping[piece.symbol()]
                if piece_channel > 0:
                    channel_index = piece_channel - 1
                else:
                    channel_index = abs(piece_channel) + 5
                
                encoded[channel_index, 7-row, col] = 1
        
        return encoded
    
    def encode_move(self, move):
        """
        Encode a chess move as a one-hot vector
        Format: from_square (6 bits) + to_square (6 bits)
        """
        from_square = move.from_square
        to_square = move.to_square
        
        move_encoding = np.zeros(64, dtype=np.float32)
        move_encoding[from_square] = 1
        move_encoding[to_square + 32] = 1
        
        return move_encoding

class ChessMovePredictor(nn.Module):
    def __init__(self):
        super(ChessMovePredictor, self).__init__()
        
        # Convolutional layers for board state feature extraction
        self.board_features = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Move prediction layers
        self.move_predictor = nn.Sequential(
            nn.Linear(256 * 8 * 8, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64)  # Output probability for each possible from-square
        )
        
        # Additional layer to predict destination square
        self.destination_predictor = nn.Sequential(
            nn.Linear(256 * 8 * 8 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 64)  # Output probability for each possible to-square
        )
    
    def forward(self, board_state):

        board_features = self.board_features(board_state)
  
        from_square_probs = self.move_predictor(board_features)
        
        combined_features = torch.cat([board_features, from_square_probs], dim=1)

        to_square_probs = self.destination_predictor(combined_features) 
        
        return from_square_probs, to_square_probs

class ChessMoveAgent:
    def __init__(self, learning_rate=0.001):
        self.feature_encoder = ChessFeatureEncoder()
        self.model = ChessMovePredictor()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def load_dataset(self, filepath):
        """Load and preprocess the chess dataset for move prediction"""
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        
        X = []
        y_from = []
        y_to = []
        
        for entry in dataset:
            board = chess.Board(entry['fen'])
        
            board_encoding = self.feature_encoder.encode_board(board)
            X.append(board_encoding)
            
            human_move = chess.Move.from_uci(entry['human_move'])
            
            from_square = human_move.from_square
            to_square = human_move.to_square
            
            y_from.append(from_square)
            y_to.append(to_square)
        
        X = torch.FloatTensor(X)       # [N, 12, 8, 8]
        y_from = torch.LongTensor(y_from)
        y_to = torch.LongTensor(y_to)
        return X, y_from, y_to
    
    def train(self, filepath, epochs=50, batch_size=32):
        X, y_from, y_to = self.load_dataset(filepath)

        X = X.to(self.device)
        y_from = y_from.to(self.device)
        y_to = y_to.to(self.device)
        
        for epoch in range(epochs):
            indices = torch.randperm(X.size(0), device=self.device)
            X_shuffled = X[indices]
            y_from_shuffled = y_from[indices]
            y_to_shuffled = y_to[indices]
            
            for i in range(0, X.size(0), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y_from = y_from_shuffled[i:i+batch_size]
                batch_y_to = y_to_shuffled[i:i+batch_size]
                
                self.optimizer.zero_grad()
                
                from_square_probs, to_square_probs = self.model(batch_X)
                
                loss_from = self.criterion(from_square_probs, batch_y_from)
                loss_to = self.criterion(to_square_probs, batch_y_to)
                loss = loss_from + loss_to
                
                loss.backward()
                
                self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    def predict_move(self, board):
        """Predict the best move for a given board position"""
        board_encoding = self.feature_encoder.encode_board(board)
        board_tensor = torch.FloatTensor(board_encoding).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            from_square_probs, to_square_probs = self.model(board_tensor)

        from_square_probs = from_square_probs.cpu().numpy()[0]
        to_square_probs = to_square_probs.cpu().numpy()[0]

        from_square = np.argmax(from_square_probs)
        to_square = np.argmax(to_square_probs)
        
        predicted_move = chess.Move(from_square, to_square)
        
        return predicted_move