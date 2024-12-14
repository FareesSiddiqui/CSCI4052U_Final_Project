import chess
import random
from colorama import Fore, Style

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def best_child(self, exploration_weight=1.4):
        """Select the best child based on UCB1 (Upper Confidence Bound)."""
        def ucb1(child):
            exploitation = child.value / (child.visits + 1e-6)
            exploration = exploration_weight * ((2 * (self.visits + 1e-6)) ** 0.5 / (child.visits + 1e-6))
            return exploitation + exploration

        return max(self.children, key=ucb1)

    def __repr__(self):
        return f"MCTSNode(move={self.move}, visits={self.visits}, value={self.value})"


def mcts_search(root, simulations=200):
    """Perform MCTS from the given root node."""
    for _ in range(simulations):
        node = root
        # Selection: Traverse the tree to find the most promising unexplored node
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        
        # Expansion: Expand the current node by adding a random unexplored child
        if not node.is_fully_expanded():
            move = random.choice([m for m in node.board.legal_moves if all(c.move != m for c in node.children)])
            child_board = node.board.copy()
            child_board.push(move)
            child_node = MCTSNode(child_board, parent=node, move=move)
            node.children.append(child_node)
            node = child_node
        
        # Simulation: Simulate a random playout from the current position
        result = simulate_random_playout(node.board)
        
        # Backpropagation: Update the node and its ancestors with the result
        backpropagate(node, result)

    return root.best_child(exploration_weight=0)  # Return the child with the highest visit count


def simulate_random_playout(board):
    """Simulate a random game from the given position and return the result."""
    temp_board = board.copy()
    while not temp_board.is_game_over():
        move = random.choice(list(temp_board.legal_moves))
        temp_board.push(move)
    if temp_board.is_checkmate():
        return 1 if temp_board.turn == chess.BLACK else -1
    else:
        return 0  # Draw


def backpropagate(node, result):
    """Propagate the simulation result up the tree."""
    while node is not None:
        node.visits += 1
        node.value += result
        result = -result  # Alternate the result for the opponent
        node = node.parent


def print_board_with_labels(board):
    """Print the chess board with ranks on the left and files on the top."""
    board_str = str(board).split("\n")
    files = "  " + " ".join([Fore.RED + char + Style.RESET_ALL for char in "abcdefgh"])

    print("\nCurrent Board:")
    print(files)  # Print file labels
    for i, row in enumerate(board_str):
        print(f"{8 - i} {row}")  # Print rank on the left side followed by the board row
    print("\n")  # Add a blank line for better readability


def main():
    # Initialize a chess board
    board = chess.Board()

    print("Welcome to the Chess Environment with MCTS!")
    print("Type 'exit' to quit.")
    print_board_with_labels(board)  # Print the starting position with labels

    while not board.is_game_over():
        if board.turn:  # White's turn (human player)
            print("\nLegal moves:", [move.uci() for move in board.legal_moves])
            move_input = input("\nEnter your move (e.g., e2e4): ").strip()
            if move_input.lower() == 'exit':
                print("Exiting the game.")
                break

            try:
                move = chess.Move.from_uci(move_input)
                if move in board.legal_moves:
                    board.push(move)
                    print_board_with_labels(board)  # Update and print the board after the move
                else:
                    print("Illegal move! Try again.")
            except ValueError:
                print("Invalid input. Please enter moves in UCI format (e.g., e2e4).")
        else:  # Black's turn (AI with MCTS)
            print("\nAI is thinking...")
            root = MCTSNode(board)
            best_child = mcts_search(root, simulations=100)
            board.push(best_child.move)
            print(f"\nAI played: {best_child.move}")
            print_board_with_labels(board)

    # Game over message
    if board.is_game_over():
        print("\nGame over!")
        if board.is_checkmate():
            print("Checkmate!")
        elif board.is_stalemate():
            print("Stalemate!")
        elif board.is_insufficient_material():
            print("Draw due to insufficient material!")
        elif board.is_seventyfive_moves():
            print("Draw due to the 75-move rule!")
        elif board.is_fivefold_repetition():
            print("Draw due to fivefold repetition!")
        else:
            print("Draw!")

if __name__ == "__main__":
    main()
