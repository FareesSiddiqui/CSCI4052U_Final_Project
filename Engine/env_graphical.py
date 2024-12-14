import pygame
import chess
import random
import torch
import numpy as np
import subprocess
import sys
import threading
from ChessPredictorClass import ChessMovePredictor

pygame.init()

SCREEN_SIZE = 640
SQUARE_SIZE = SCREEN_SIZE // 8
LIGHT_COLOR = (240, 217, 181)
DARK_COLOR = (181, 136, 99)
PIECE_SPRITES = pygame.image.load("chess_pieces.png")

FONT = pygame.font.Font(None, 36)
TITLE_FONT = pygame.font.Font(None, 48)
BUTTON_FONT = pygame.font.Font(None, 36)

# Shared variable for fen updates from CV model
fen_from_cv = None
fen_lock = threading.Lock()

def get_piece_image(piece):
    sprite_width = PIECE_SPRITES.get_width() // 6
    sprite_height = PIECE_SPRITES.get_height() // 2
    piece_map = {
        "K": (0, 0), "Q": (1, 0), "B": (2, 0), "N": (3, 0), "R": (4, 0), "P": (5, 0),
        "k": (0, 1), "q": (1, 1), "b": (2, 1), "n": (3, 1), "r": (4, 1), "p": (5, 1),
    }
    col, row = piece_map[piece.symbol()]
    rect = pygame.Rect(col * sprite_width, row * sprite_height, sprite_width, sprite_height)
    sprite = PIECE_SPRITES.subsurface(rect).convert_alpha()
    return pygame.transform.scale(sprite, (SQUARE_SIZE, SQUARE_SIZE))

def draw_board(screen, board, selected_square=None):
    small_font = pygame.font.Font(None, 24)

    for row in range(8):
        for col in range(8):
            color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            if selected_square is not None and chess.square(col, 7 - row) == selected_square:
                overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                overlay.set_alpha(200)
                overlay.fill((173, 216, 230))
                screen.blit(overlay, (col * SQUARE_SIZE, row * SQUARE_SIZE))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            piece_image = get_piece_image(piece)
            screen.blit(piece_image, (col * SQUARE_SIZE, row * SQUARE_SIZE))

    # Rank and file labels
    for row in range(8):
        square_color = LIGHT_COLOR if (0 + row) % 2 == 0 else DARK_COLOR
        label_color = DARK_COLOR if square_color == LIGHT_COLOR else LIGHT_COLOR
        rank_label = small_font.render(str(8 - row), True, label_color)
        rank_x = 5
        rank_y = row * SQUARE_SIZE + 5
        screen.blit(rank_label, (rank_x, rank_y))

    for col in range(8):
        square_color = LIGHT_COLOR if (col + 7) % 2 == 0 else DARK_COLOR
        label_color = DARK_COLOR if square_color == LIGHT_COLOR else LIGHT_COLOR
        file_label = small_font.render(chr(ord('a') + col), True, label_color)
        file_x = (col + 1) * SQUARE_SIZE - 15
        file_y = SCREEN_SIZE - SQUARE_SIZE + 5
        screen.blit(file_label, (file_x, file_y))

def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            tensor[piece_map[piece.symbol()], row, col] = 1.0
    return tensor

def predict_move(model, board, device, top_k=10):
    board_tensor = torch.tensor(fen_to_tensor(board.fen()), dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        from_square_probs, to_square_probs = model(board_tensor)
        from_square_probs = from_square_probs.squeeze().cpu().numpy()
        to_square_probs = to_square_probs.squeeze().cpu().numpy()

    from_square_indices = np.argsort(from_square_probs)[::-1][:top_k]
    to_square_indices = np.argsort(to_square_probs)[::-1][:top_k]

    candidate_moves = [
        chess.Move(from_square, to_square)
        for from_square in from_square_indices
        for to_square in to_square_indices
    ]
    return candidate_moves

def select_promotion(screen, square, color):
    pieces = ['Q', 'R', 'B', 'N']
    piece_images = [get_piece_image(chess.Piece.from_symbol(piece.lower() if color == chess.BLACK else piece)) for piece in pieces]
    menu_width = SQUARE_SIZE * len(pieces)
    menu_height = SQUARE_SIZE
    x_start = (square % 8) * SQUARE_SIZE
    y_start = (7 - (square // 8)) * SQUARE_SIZE if color == chess.WHITE else (square // 8) * SQUARE_SIZE

    menu_rect = pygame.Rect(x_start, y_start, menu_width, menu_height)
    pygame.draw.rect(screen, (50, 50, 50), menu_rect)

    for i, img in enumerate(piece_images):
        screen.blit(img, (x_start + i * SQUARE_SIZE, y_start))

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                if menu_rect.collidepoint(mouse_x, mouse_y):
                    index = (mouse_x - x_start) // SQUARE_SIZE
                    return pieces[index]

def draw_home_screen(screen):
    screen.fill((0, 0, 0))
    title_text = TITLE_FONT.render("Chess Project", True, (255, 255, 255))
    engine_text = BUTTON_FONT.render("Play Engine", True, (255, 255, 255))
    otb_text = BUTTON_FONT.render("Over The Board Analysis", True, (255, 255, 255))

    title_rect = title_text.get_rect(center=(SCREEN_SIZE // 2, SCREEN_SIZE // 4))
    screen.blit(title_text, title_rect)

    engine_rect = engine_text.get_rect(center=(SCREEN_SIZE // 2, SCREEN_SIZE // 2))
    screen.blit(engine_text, engine_rect)

    otb_rect = otb_text.get_rect(center=(SCREEN_SIZE // 2, SCREEN_SIZE // 2 + 60))
    screen.blit(otb_text, otb_rect)

    return engine_rect, otb_rect

def run_engine_mode(screen, model, device):
    board = chess.Board()
    running = True
    selected_square = None
    ai_thinking = False

    while running:
        screen.fill((0, 0, 0))
        draw_board(screen, board, selected_square)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.MOUSEBUTTONDOWN and board.turn and not ai_thinking:
                mouse_x, mouse_y = event.pos
                col = mouse_x // SQUARE_SIZE
                row = 7 - (mouse_y // SQUARE_SIZE)
                square = chess.square(col, row)
                if selected_square is None:
                    if board.piece_at(square) and board.piece_at(square).color == board.turn:
                        selected_square = square
                else:
                    move = chess.Move(from_square=selected_square, to_square=square)
                    if (chess.square_rank(move.to_square) in [0,7] and 
                        board.piece_at(selected_square) and 
                        board.piece_at(selected_square).piece_type == chess.PAWN):
                        promotion_piece = select_promotion(screen, move.to_square, board.turn)
                        move.promotion = chess.Piece.from_symbol(promotion_piece.lower()).piece_type

                    if move in board.legal_moves:
                        board.push(move)
                        selected_square = None

                        if board.is_checkmate():
                            print("Checkmate! Player wins!")
                            running = False
                        elif board.is_stalemate():
                            print("Stalemate! The game is a draw.")
                            running = False

                        ai_thinking = True
                        pygame.time.set_timer(pygame.USEREVENT + 1, 500)
                    else:
                        selected_square = None

            if event.type == pygame.USEREVENT + 1 and not board.turn:
                pygame.time.set_timer(pygame.USEREVENT + 1, 0)
                candidate_moves = predict_move(model, board, device, top_k=10)
                valid_move = None

                for move in candidate_moves:
                    if move in board.legal_moves:
                        valid_move = move
                        break

                if not valid_move:
                    valid_move = random.choice(list(board.legal_moves))
                    print("Engine failed to find a valid move from predictions. Playing a random move.")

                board.push(valid_move)
                print(f"Engine move: {valid_move}")

                if board.is_checkmate():
                    print("Checkmate! Engine wins!")
                    running = False
                elif board.is_stalemate():
                    print("Stalemate! The game is a draw.")
                    running = False

                ai_thinking = False

        pygame.display.flip()

def read_fen_output(process):
    global fen_from_cv
    # Continuously read lines from the subprocess stdout
    for line in process.stdout:
        line = line.strip()
        print(line)  # Debug print
        if "Generated FEN:" in line:
            # Extract fen
            parts = line.split("Generated FEN:")
            if len(parts) > 1:
                fen_str = parts[1].strip()
                with fen_lock:
                    fen_from_cv = fen_str

def run_otb_mode(screen):
    global fen_from_cv

    # Start with an empty board
    board = chess.Board(None)
    board.clear()

    # Show the empty board
    screen.fill((0,0,0))
    draw_board(screen, board, None)
    pygame.display.flip()

    # Run the CV model in parallel
    cmd = ["python", "..\\BoardParserV2\\yolov5\\detect.py", 
           "--weights", "..\\BoardParserV2\\yolov5\\runs\\train\\exp6\\weights\\best.pt", 
           "--source", "0"]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Start a thread to read output
    thread = threading.Thread(target=read_fen_output, args=(process,), daemon=True)
    thread.start()

    running = True
    selected_square = None

    while running:
        screen.fill((0, 0, 0))

        # Check if we have a new fen
        with fen_lock:
            if fen_from_cv is not None:
                try:
                    new_board = chess.Board(fen_from_cv)
                    board = new_board
                except ValueError:
                    print("Invalid FEN received, ignoring...")
                fen_from_cv = None

        draw_board(screen, board, selected_square)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Allow user to move pieces manually
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                col = mouse_x // SQUARE_SIZE
                row = 7 - (mouse_y // SQUARE_SIZE)
                square = chess.square(col, row)
                if selected_square is None:
                    if board.piece_at(square) and board.piece_at(square).color == board.turn:
                        selected_square = square
                else:
                    move = chess.Move(from_square=selected_square, to_square=square)
                    # Auto-promote to queen for simplicity
                    if (chess.square_rank(move.to_square) in [0,7] and 
                        board.piece_at(selected_square) and
                        board.piece_at(selected_square).piece_type == chess.PAWN):
                        move.promotion = chess.QUEEN

                    if move in board.legal_moves:
                        board.push(move)
                    selected_square = None

        pygame.display.flip()

    process.terminate()  # Clean up if needed

def main():
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Chess")

    model_path = "chess_move_predictor.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessMovePredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    state = 'home'
    running = True

    while running:
        if state == 'home':
            engine_rect, otb_rect = draw_home_screen(screen)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if engine_rect.collidepoint(mx, my):
                        state = 'engine_mode'
                    elif otb_rect.collidepoint(mx, my):
                        state = 'otb_mode'

        elif state == 'engine_mode':
            run_engine_mode(screen, model, device)
            state = 'home'

        elif state == 'otb_mode':
            run_otb_mode(screen)
            state = 'home'

    pygame.quit()

if __name__ == "__main__":
    main()
