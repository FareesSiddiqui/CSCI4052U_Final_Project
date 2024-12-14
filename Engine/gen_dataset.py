import chess
import chess.pgn
import chess.engine

import glob
import json
import os

# dataset https://storage.lczero.org/files/training_data/
## DO NOT REMOVE ABOVE COMMENT, IT'S WHERE I GOT MY DATA, MIGHT NEED TO CITE IT

STOCKFISH_PATH = "stockfish/engine"  
PGN_DIR = "Lichess_Database"    
OUTPUT_JSON = "lichess_dataset_combined.json"
DEPTH = 10
LIMIT_POSITIONS = 100000  
TIME_PER_POSITION = 0.1

def process_pgn_file(file_path, engine, dataset, limit_positions=None, depth=10, time_per_position=0.1):
    positions_count = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break  

            board = game.board()
            for move in game.mainline_moves():
                fen_before = board.fen()
                human_move_uci = move.uci()

                info = engine.analyse(board, limit=chess.engine.Limit(time=time_per_position, depth=depth))
                sf_score = info["score"].white()
                if sf_score.is_mate():
                    mate_in = sf_score.mate()
                    eval_cp = 10000 if mate_in > 0 else -10000
                else:
                    eval_cp = sf_score.score(mate_score=10000)

                result = engine.play(board, limit=chess.engine.Limit(time=time_per_position, depth=depth))
                sf_best_move_uci = result.move.uci() if result.move else human_move_uci

                dataset.append({
                    "fen": fen_before,
                    "human_move": human_move_uci,
                    "stockfish_best_move": sf_best_move_uci,
                    "stockfish_eval": eval_cp
                })

                positions_count += 1
                if limit_positions is not None and positions_count >= limit_positions:
                    return positions_count

                board.push(move)

    return positions_count

def process_all_pgn_files(pgn_dir, output_json, limit_positions=None, depth=10, time_per_position=0.1):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    dataset = []
    total_positions = 0

    pgn_files = glob.glob(os.path.join(pgn_dir, "*.pgn"))
    total_files = len(pgn_files)

    for i, pgn_file in enumerate(pgn_files, start=1):
        remaining_limit = None if limit_positions is None else (limit_positions - total_positions)
        file_positions = process_pgn_file(
            pgn_file,
            engine,
            dataset,
            limit_positions=remaining_limit,
            depth=depth,
            time_per_position=time_per_position
        )

        if file_positions is None:
            file_positions = 0

        total_positions += file_positions
        print(f"Processed {file_positions} from file {i}/{total_files} | File: {pgn_file}")

        if limit_positions is not None and total_positions >= limit_positions:
            break

    engine.quit()

    with open(output_json, "w") as out_file:
        json.dump(dataset, out_file, indent=2)

    print(f"Processed {total_positions} positions from {min(i, total_files)} files total and saved to {output_json}.")

if __name__ == "__main__":
    process_all_pgn_files(
        pgn_dir=PGN_DIR,
        output_json=OUTPUT_JSON,
        limit_positions=LIMIT_POSITIONS,
        depth=DEPTH,
        time_per_position=TIME_PER_POSITION
    )