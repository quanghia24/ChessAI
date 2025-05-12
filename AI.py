import random
from math import inf
from piece import *
import time

PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

KING_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

PIECE_SQUARE_TABLES = {
    Pawn: PAWN_TABLE,
    Knight: KNIGHT_TABLE,
    Bishop: BISHOP_TABLE,
    Rook: ROOK_TABLE,
    Queen: QUEEN_TABLE,
    King: KING_TABLE
}

def random_move(board):
    """
    Selects a random move from the valid moves for the current players turn
    :param board: the current board being used for the game (Board)
    :return: tuple representing move; format: ((sourceX, sourceY), (destX, destY))
    """
    moves = board.get_moves()
    if moves:
        return random.choice(moves)

def positional_score(board, color):
    score = 0
    for row in board.tilemap:
        for tile in row:
            piece = tile.piece
            if piece and piece.color == color:
                table = PIECE_SQUARE_TABLES.get(type(piece))
                if table:
                    x, y = tile.x, tile.y
                    # Flip vertically for black
                    table_index = (7 - y) * 8 + x if color == BLACK else y * 8 + x
                    score += table[table_index]

                    # Penalize knights in dangerous positions
                    if isinstance(piece, Knight):
                        attackers = board.get_attackers(x, y, BLACK if color == WHITE else WHITE)
                        if attackers:  # if any opponent can attack this square
                            score -= len(attackers)*5  # apply penalty for knight being in a dangerous square
    return score

"""Mobility helps avoid cramped positions."""
def mobility_score(board, color):
    original_turn = board.turn
    board.turn = color
    mobility = len(board.get_moves())

    # Add a penalty for moving to a threatened square
    danger_penalty = 0
    for row in board.tilemap:
        for tile in row:
            piece = tile.piece
            if piece and piece.color == color and (isinstance(piece, Knight) or isinstance(piece, Bishop) or isinstance(piece, Rook) or isinstance(piece, Queen)):
                attackers = board.get_attackers(piece.x, piece.y, BLACK if color == WHITE else WHITE)
                for attacker in attackers:
                    danger_penalty += 2  # Increase penalty for each threat to the knight


    board.turn = original_turn
    return mobility - danger_penalty # Tune this weight

def threat_penalty(board, color):
    """
    Evaluate the threats against pieces of the given color.
    :param board: the current board
    :param color: the color to check for threatened pieces
    :return: penalty score for threatened pieces
    """
    penalty = 0
    opposite_color = BLACK if color == WHITE else WHITE
    for row in board.tilemap:
        for tile in row:
            piece = tile.piece
            if piece and piece.color == color:
                attackers = board.get_attackers(piece.x, piece.y, opposite_color)
                if attackers:
                    defenders = board.get_attackers(piece.x, piece.y, color)
                    if len(attackers) > len(defenders):
                        if defenders != []:
                            penalty += board.weights[type(piece)]
                        else:
                            penalty += board.weights[type(piece)] * 1.5
    return penalty

def is_defended(board, piece):
    defenders = board.get_attackers(piece.x, piece.y, piece.color)
    if defenders != []:
        return True
    return False

def promotion_potential(board, color):
    bonus = 0
    for row in board.tilemap:
        for tile in row:
            piece = tile.piece
            if isinstance(piece, Pawn) and piece.color == color:
                if (color == WHITE and tile.y >= 6) or (color == BLACK and tile.y <= 1):
                    bonus += 20  
    return bonus

def development_score(board, color):
    score = 0
    for row in board.tilemap:
        for tile in row:
            piece = tile.piece
            if piece and piece.color == color and isinstance(piece, (Knight, Bishop)):
                if (piece.x, piece.y) != piece.start_square:
                    score += 5  # reward piece being moved off the back rank
    return score

def king_safety(board, color):
    king_pos = board.whiteKingCoords if color == WHITE else board.blackKingCoords
    attackers = board.get_attackers(*king_pos, BLACK if color == WHITE else WHITE)
    return -len(attackers) * 40  # Penalize exposed kings

def rook_connected(board, color):
    back_rank = 0 if color == WHITE else 7
    rook_files = [x for x in range(8) if isinstance(board.tilemap[x][back_rank].piece, Rook) and board.tilemap[x][back_rank].piece.color == color]
    if len(rook_files) >= 2:
        # Check if there's a path between rooks
        min_file, max_file = min(rook_files), max(rook_files)
        for x in range(min_file + 1, max_file):
            if board.tilemap[x][back_rank].piece is not None:
                return 0
        return 10  # Bonus for connected rooks
    return 0
"""
- Multiple attackers on the same square (focus/pressure)
- Pins and skewers (optional, advanced)
- Pressure on valuable targets (e.g., attacking a queen with two pieces)
"""
def attack_coordination_score(board, color):
    score = 0
    target_map = {}
 
    # Step 1: Scan all squares and map how many times each is attacked by `color`
    for row in range(8):
        for col in range(8):
            attackers = board.get_attackers(col, row, color)
            if attackers:
                target_map[(col, row)] = attackers

    for (x, y), attackers in target_map.items():
        if len(attackers) >= 2:
            target_tile = board.tilemap[x][y]
            target_piece = target_tile.piece

            # Reward for coordination (attacking same square with multiple pieces)
            score += 5 * (len(attackers) - 1)

            if target_piece and target_piece.color != color:
                # Bonus for coordinating attack on high-value target
                score += board.weights.get(type(target_piece), 0) * (len(attackers) - 1)

    return score

"""
1. Pawn structure & development:
The AI treats pawns mostly as material, not positionally strategic pieces. It doesn't understand:
    Advancing pawns for space
    Supporting other pawns (chains)
    Opening lanes for bishops/rooks/queen
    Central control
2. Attacking plans / tactics:
The AI can't “plan” or recognize:
    Pawn breaks
    Opening lines
    King safety deterioration
    Attacking weak pawns or back rank issues
"""
def pawn_structure_score(board, color):
    score = 0
    files = {i: [] for i in range(8)}  # map of x (file) -> list of pawn y (rank)
    
    for x in range(8):
        for y in range(8):
            piece = board.tilemap[x][y].piece
            if piece and isinstance(piece, Pawn) and piece.color == color:
                files[x].append(y)

                # Central pawns get bonus
                if x in [3, 4] and (y > 1 and y < 6):
                    score += 3

                # Passed pawn (no enemy pawns on file or adjacent)
                enemy_color = BLACK if color == WHITE else WHITE
                is_passed = True
                for dx in [x-1, x, x+1]:
                    if 0 <= dx < 8:
                        for dy in range(8):
                            enemy_piece = board.tilemap[dx][dy].piece
                            if enemy_piece and isinstance(enemy_piece, Pawn) and enemy_piece.color == enemy_color:
                                is_passed = False
                if is_passed:
                    score += 10

    # Penalize doubled pawns
    for x in files:
        if len(files[x]) > 1:
            score -= 5 * (len(files[x]) - 1)

    return score

def active_piece_count(board, color):
    active = 0
    for row in board.tilemap:
        for tile in row:
            piece = tile.piece
            if piece and piece.color == color and (piece.x, piece.y) != piece.start_square:
                active += 1
    return active

def early_king_move_penalty(board, color):
    king_pos = board.whiteKingCoords if color == WHITE else board.blackKingCoords
    back_rank = 0 if color == WHITE else 7
    if king_pos[1] != back_rank:
        return -50  # King has moved forward early
    return 0

def early_queen_penalty(board, color):
    penalty = 0
    queen_start_y = 0 if color == BLACK else 7
    for row in board.tilemap:
        for tile in row:
            piece = tile.piece
            if isinstance(piece, Queen) and piece.color == color:
                if tile.y != queen_start_y:
                    # Queen has moved early
                    if len(board.past_moves) < 12:
                        penalty -= 25  
    return penalty

def knight_aggression_penalty(board, color):
    penalty = 0
    enemy_color = BLACK if color == WHITE else WHITE
    for row in board.tilemap:
        for tile in row:
            piece = tile.piece
            if isinstance(piece, Knight) and piece.color == color:
                # Penalize knights that are too deep in enemy territory
                if (color == WHITE and piece.y >= 5) or (color == BLACK and piece.y <= 4):
                    attackers = board.get_attackers(piece.x, piece.y, enemy_color)
                    if attackers:  # Penalize if knight is under attack
                        penalty += 10
    return penalty

def overextension_penalty(board, color):
    penalty = 0
    enemy_color = BLACK if color == WHITE else WHITE

    for row in board.tilemap:
        for tile in row:
            piece = tile.piece
            if piece and piece.color == color and isinstance(piece, (Knight, Bishop)):
                defenders = board.get_attackers(piece.x, piece.y, color)
                attackers = board.get_attackers(piece.x, piece.y, enemy_color)
                if attackers and len(attackers) > len(defenders):
                    penalty += board.weights.get(type(piece), 0) // 2  # penalize dangerous position
                if piece.y < 2 or piece.y > 5:  # very deep into enemy territory
                    penalty += 10
    return penalty

def safe_development_bonus(board, color):
    score = 0
    for row in board.tilemap:
        for tile in row:
            piece = tile.piece
            if piece and piece.color == color and isinstance(piece, (Knight, Bishop)):
                if is_defended(board, piece):
                    score += 5
    return score

def evaluate(board, maximizing_color):
    """
    Provides a number representing the value of the board at a given state
    :param board: the current board being used for the game (Board)
    :param maximizing_color: color associated with maximizing player (tuple)
    :return: integer representing boards value
    """
    opponent_color = BLACK if maximizing_color == WHITE else WHITE

    if board.is_checkmate():
        return float('inf') if board.turn != maximizing_color else -float('inf')
    elif board.is_stalemate():
        return 0

    material         = board.whiteScore - board.blackScore if maximizing_color == WHITE else board.blackScore - board.whiteScore
    positional       = positional_score(board, maximizing_color) - positional_score(board, opponent_color)
    mobility         = mobility_score(board, maximizing_color) - mobility_score(board, opponent_color)
    threats          = threat_penalty(board, opponent_color) - threat_penalty(board, maximizing_color)
    promotion        = promotion_potential(board, maximizing_color) - promotion_potential(board, opponent_color)
    development      = development_score(board, maximizing_color) - development_score(board, opponent_color)
    king_safety_val  = king_safety(board, maximizing_color) - king_safety(board, opponent_color)
    rook_conn        = rook_connected(board, maximizing_color) - rook_connected(board, opponent_color)
    coordination     = attack_coordination_score(board, maximizing_color) - attack_coordination_score(board, opponent_color)
    pawn_score       = pawn_structure_score(board, maximizing_color) - pawn_structure_score(board, opponent_color)
    active_bonus     = active_piece_count(board, maximizing_color)
    overext          = overextension_penalty(board, opponent_color) - overextension_penalty(board, maximizing_color)
    safe_dev         = safe_development_bonus(board, maximizing_color) - safe_development_bonus(board, opponent_color)
    early_king_move  = early_king_move_penalty(board, maximizing_color)
    early_queen_move = early_queen_penalty(board, maximizing_color)



    total_score = (2* material +
            0.4 * positional +
            0.15 * mobility +
            0.5 * threats +
            0.2 * promotion +
            0.2 * development +
            0.3 * king_safety_val +
            0.2 * rook_conn +
            0.3 * coordination +
            0.3 * pawn_score +
            0.5 * active_bonus +
            0.2 * overext +
            0.3 * safe_dev + 
            early_king_move +
            early_queen_move)
    
    if development > 20 and coordination < 5:
        total_score -= 10  # undeveloped army stuck
    early_game_bonus = 0
    if len(board.past_moves) < 10:
        early_game_bonus = 0.3 * development + 0.4 * mobility + 0.3 * pawn_score


     
    return total_score + early_game_bonus




def quiescence_search(board, alpha, beta, maximizing_color):
    stand_pat = evaluate(board, maximizing_color)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    # Only explore capture moves
    for move in board.get_moves():
        src, dst = move
        if not board.is_capture(src, dst):
            continue

        board.make_move(src, dst)
        score = -quiescence_search(board, -beta, -alpha, maximizing_color)
        board.unmake_move()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


transposition_table = {}

def minimax(board, depth, alpha, beta, maximizing_player, maximizing_color):
    """
    Minimax algorithm used to find best move for the AI
    :param board: the current board being used for the game (Board)
    :param depth: controls how deep to search the tree of possible moves (int)
    :param alpha: the best value that the maximizer currently can guarantee at that level or above (int)
    :param beta: the best value that the minimizer currently can guarantee at that level or above (int)
    :param maximizing_player: True if current player is maximizing player (bool)
    :param maximizing_color: color of the AI using this function to determine a move (tuple)
    :return: tuple representing move and eval; format: (move, eval)
    """
    if depth == 0 or board.gameover:
        return None, evaluate(board, maximizing_color)

    board_hash = board.get_hash() 
    if board_hash in transposition_table:
        entry = transposition_table[board_hash]
        if entry['depth'] >= depth: 
            return entry['move'], entry['eval']

    moves = board.get_moves_sorted()
    if not moves:
        return None, inf
    best_move = random.choice(moves)

    if maximizing_player:
        max_eval = -inf
        for move in moves:
            if move is None:
                time.sleep(5)
                print("game over or no more moves")
                return None, evaluate(board, maximizing_color)
            board.make_move(move[0], move[1])
            current_eval = minimax(board, depth-1, alpha, beta, False, maximizing_color)[1]
            # print(f"Depth {depth}: Move {move} Eval {current_eval}")
            board.unmake_move()
            if current_eval > max_eval:
                max_eval = current_eval
                best_move = move
            alpha = max(alpha, current_eval)
            if beta <= alpha:
                break
        transposition_table[board_hash] = {'depth': depth, 'eval': max_eval, 'move': best_move}
        return best_move, max_eval
    else:
        min_eval = inf
        for move in moves:
            if move is None:
                time.sleep(5)
                print("game over or no more moves")
                return None, evaluate(board, maximizing_color)
            board.make_move(move[0], move[1])
            current_eval = minimax(board, depth-1, alpha, beta, True, maximizing_color)[1]
            # print(f"Depth {depth}: Move {move} Eval {current_eval}")
            board.unmake_move()
            if current_eval < min_eval:
                min_eval = current_eval
                best_move = move
            beta = min(beta, current_eval)
            if beta <= alpha:
                break
        transposition_table[board_hash] = {'depth': depth, 'eval': min_eval, 'move': best_move}
        return best_move, min_eval




