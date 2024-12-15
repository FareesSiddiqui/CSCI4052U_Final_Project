import chess
import chess.engine
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class ChessEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, stockfish_path="stockfish/engine", time_limit=0.1, max_episode_length=100):
        super(ChessEnv, self).__init__()

        self.board = chess.Board()
        self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.stockfish_time_limit = time_limit
        self.max_episode_length = max_episode_length
        self.episode_steps = 0

        self.observation_space = spaces.Box(low=0, high=1, shape=(12, 8, 8), dtype=np.float32)

        self.action_space = spaces.Discrete(64 * 64)

    def reset(self, seed=None, options=None):
        """Reset the board and return the initial observation."""
        super().reset(seed=seed)
        self.board.reset()
        self.episode_steps = 0
        return self.encode_board(), {}

    def step(self, action):
        """Take an action and return the new state, reward, terminated, truncated, and info."""
        from_square = action // 64
        to_square = action % 64
        move = chess.Move(from_square, to_square)

        if move in self.board.legal_moves:
            self.board.push(move)
            reward = self.calculate_reward()
            terminated = self.board.is_game_over()
            truncated = False
        else:
            reward = -10
            terminated = True
            truncated = False

        if not terminated:
            result = self.stockfish.play(self.board, chess.engine.Limit(time=self.stockfish_time_limit))
            self.board.push(result.move)
            reward += self.calculate_reward()
            terminated = self.board.is_game_over()

        self.episode_steps += 1
        if self.episode_steps >= self.max_episode_length:
            truncated = True

        return self.encode_board(), reward, terminated, truncated, {}

    def encode_board(self):
        """Encode the chess board as a 12x8x8 tensor."""
        piece_map = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
        }
        encoded = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                channel = piece_map[piece.symbol()]
                channel_idx = channel - 1 if channel > 0 else abs(channel) + 5
                row, col = divmod(square, 8)
                encoded[channel_idx, row, col] = 1
        return encoded

    def calculate_reward(self):
        """Calculate reward based on material advantage."""
        material_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
        material = sum(
            material_values[piece.symbol().upper()] if piece.color == chess.WHITE else -material_values[piece.symbol().upper()]
            for piece in self.board.piece_map().values()
        )
        return material / 100

    def render(self, mode="human"):
        """Render the board."""
        print(self.board)

    def close(self):
        """Close the Stockfish engine."""
        self.stockfish.quit()


env = ChessEnv(stockfish_path="stockfish/engine")
vec_env = make_vec_env(lambda: env, n_envs=1)
model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")
model.learn(total_timesteps=100000, progress_bar=True)
model.save("chess_ppo_agent")
env.close()