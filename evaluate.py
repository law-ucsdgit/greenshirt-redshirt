from __future__ import annotations

import argparse
import json
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request

from connect4.game import ACTION_SPACE
from connect4.model import DQN

app = Flask(__name__)


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Connect Four vs Checkpoint</title>
  <style>
    :root {
      --board-blue: #0d4ed1;
      --board-bg: #eaf2ff;
      --player-1: #ff4d4d;
      --player-2: #ffd633;
      --empty: #f8f9fb;
      --text: #11263c;
      --win: #1ec96b;
    }
    body {
      font-family: Arial, sans-serif;
      background: linear-gradient(135deg, #eef5ff, #dfeeff);
      color: var(--text);
      margin: 0;
      padding: 24px;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }
    .container {
      background: rgba(255,255,255,0.8);
      border-radius: 18px;
      box-shadow: 0 18px 40px rgba(17, 38, 60, 0.12);
      padding: 24px;
      max-width: 820px;
      width: 100%;
      text-align: center;
    }
    h1 {
      margin-top: 0;
    }
    .status {
      margin-bottom: 18px;
      font-size: 1.1rem;
      font-weight: bold;
      min-height: 28px;
    }
    .board {
      display: grid;
      grid-template-columns: repeat(7, minmax(42px, 1fr));
      gap: 10px;
      background: var(--board-blue);
      padding: 14px;
      border-radius: 18px;
      max-width: 660px;
      margin: 0 auto;
    }
    .cell {
      width: 100%;
      aspect-ratio: 1;
      border-radius: 50%;
      background: var(--empty);
      border: 2px solid rgba(17,38,60,0.08);
      cursor: pointer;
      transition: transform 0.12s ease, box-shadow 0.12s ease;
      box-sizing: border-box;
    }
    .cell:hover {
      transform: scale(1.04);
      box-shadow: inset 0 0 0 2px rgba(13,78,209,0.25);
    }
    .cell.player-1 { background: var(--player-1); }
    .cell.player-2 { background: var(--player-2); }
    .cell.win { box-shadow: 0 0 0 3px var(--win); }
    .controls {
      margin-top: 16px;
    }
    button {
      background: var(--board-blue);
      color: white;
      border: none;
      border-radius: 10px;
      padding: 10px 18px;
      font-size: 1rem;
      cursor: pointer;
    }
    button:hover { opacity: 0.95; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Connect Four vs Checkpoint</h1>
    <div id="status" class="status">Your turn — pick a column.</div>
    <div id="board" class="board" aria-label="Connect Four Board"></div>
    <div class="controls">
      <button id="reset">New game</button>
    </div>
  </div>

  <script>
    const boardEl = document.getElementById('board');
    const statusEl = document.getElementById('status');
    const resetEl = document.getElementById('reset');
    const rows = 6;
    const cols = 7;
    let board = Array.from({ length: rows }, () => Array(cols).fill(0));
    let currentPlayer = 1;
    let gameOver = false;
    let waitingForAi = false;

    function renderBoard() {
      boardEl.innerHTML = '';
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const cell = document.createElement('div');
          cell.className = 'cell';
          const value = board[r][c];
          if (value === 1) cell.classList.add('player-1');
          if (value === 2) cell.classList.add('player-2');
          cell.dataset.row = String(r);
          cell.dataset.col = String(c);
          cell.addEventListener('click', () => handleHumanMove(c));
          boardEl.appendChild(cell);
        }
      }
    }

    function legalMoves() {
      return Array.from({ length: cols }, (_, c) => c).filter((c) => board[0][c] === 0);
    }

    function dropToken(column, player) {
      for (let r = rows - 1; r >= 0; r--) {
        if (board[r][column] === 0) {
          board[r][column] = player;
          return { row: r, col: column };
        }
      }
      return null;
    }

    function checkWinner(boardState, player) {
      const dirs = [
        [0, 1],
        [1, 0],
        [1, 1],
        [1, -1],
      ];

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          if (boardState[r][c] !== player) continue;
          for (const [dr, dc] of dirs) {
            let count = 1;
            for (let step = 1; step < 4; step++) {
              const rr = r + dr * step;
              const cc = c + dc * step;
              if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) break;
              if (boardState[rr][cc] !== player) break;
              count += 1;
            }
            if (count >= 4) return true;
          }
        }
      }
      return false;
    }

    function setStatus(message) {
      statusEl.textContent = message;
    }

    async function requestAiMove() {
      waitingForAi = true;
      setStatus('Checkpoint is thinking...');
      const response = await fetch('/api/ai-move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board: board, player: 2 })
      });
      const result = await response.json();

      if (!result.ok) {
        waitingForAi = false;
        setStatus(result.error || 'Unable to compute move.');
        return;
      }

      const column = result.column;
      dropToken(column, 2);
      waitingForAi = false;
      if (checkWinner(board, 2)) {
        gameOver = true;
        setStatus('Checkpoint wins!');
        renderBoard();
        return;
      }

      currentPlayer = 1;
      setStatus('Your turn — pick a column.');
      renderBoard();
    }

    function handleHumanMove(column) {
      if (gameOver || waitingForAi || currentPlayer !== 1) return;
      if (board[0][column] !== 0) return;

      const move = dropToken(column, 1);
      if (move === null) return;

      if (checkWinner(board, 1)) {
        gameOver = true;
        setStatus('You win!');
        renderBoard();
        return;
      }

      currentPlayer = 2;
      renderBoard();
      requestAiMove();
    }

    function resetGame() {
      board = Array.from({ length: rows }, () => Array(cols).fill(0));
      currentPlayer = 1;
      gameOver = false;
      waitingForAi = false;
      setStatus('Your turn — pick a column.');
      renderBoard();
    }

    resetEl.addEventListener('click', resetGame);
    renderBoard();
  </script>
</body>
</html>
"""


def _first_linear_dim(state_dict: dict) -> Optional[int]:
    for value in state_dict.values():
        if hasattr(value, "shape") and len(value.shape) >= 2:
            return int(value.shape[-1])
    return None


def _board_to_model_tensor(board: List[List[int]], input_dim: int) -> torch.Tensor:
    flat = np.asarray(board, dtype=np.float32).reshape(-1)
    if input_dim == 49:
        padded = np.zeros((7, 7), dtype=np.float32)
        padded[1:, :] = np.asarray(board, dtype=np.float32)
        flat = padded.reshape(-1)
    if flat.size != input_dim:
        raise ValueError(f"Board size mismatch: expected {input_dim}, got {flat.size}")
    return torch.tensor(flat, dtype=torch.float32).unsqueeze(0)


def _legal_moves(board: List[List[int]]) -> List[int]:
    return [col for col in range(len(board[0])) if board[0][col] == 0]


def _copy_board(board: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in board]


def _drop_token(board: List[List[int]], column: int, player: int) -> Optional[Tuple[int, int]]:
    for row in range(len(board) - 1, -1, -1):
        if board[row][column] == 0:
            board[row][column] = player
            return row, column
    return None


def _check_winner(board: List[List[int]], player: int) -> bool:
    rows = len(board)
    cols = len(board[0])
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for r in range(rows):
        for c in range(cols):
            if board[r][c] != player:
                continue
            for dr, dc in directions:
                count = 1
                for step in range(1, 4):
                    rr = r + dr * step
                    cc = c + dc * step
                    if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                        break
                    if board[rr][cc] != player:
                        break
                    count += 1
                if count >= 4:
                    return True
    return False


def load_model(checkpoint_path: str) -> Tuple[DQN, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    input_dim = _first_linear_dim(state_dict)
    if input_dim is None:
        raise ValueError("Could not infer the model input size from the checkpoint.")
    model = DQN(input_dim, 7)
    model.load_state_dict(state_dict)
    model.eval()
    return model, input_dim


def choose_action(model: DQN, board: List[List[int]], input_dim: int, ai_player: int = 2) -> int:
    valid_cols = _legal_moves(board)
    if not valid_cols:
        raise ValueError("No legal moves available.")

    best_col = valid_cols[0]
    best_value = float("-inf")
    for col in valid_cols:
        next_board = _copy_board(board)
        move = _drop_token(next_board, col, ai_player)
        if move is None:
            continue
        if _check_winner(next_board, ai_player):
            return col
        tensor = _board_to_model_tensor(next_board, input_dim)
        with torch.no_grad():
            logits = model(tensor)
        value = logits[0, col].item()
        if value > best_value:
            best_value = value
            best_col = col
    return best_col


def evaluate(checkpoint_path: str, episodes: int = 10):
    env = __import__("connect4.game", fromlist=["rlEnvironment"]).rlEnvironment()
    model, _ = load_model(checkpoint_path)

    wins = 0
    losses = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            board = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action = model(board.view(1, -1)).max(1).indices.item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1

    print(f"Wins: {wins} | Losses: {losses} | Games: {episodes}")


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/api/ai-move", methods=["POST"])
def api_ai_move():
    payload = request.get_json(force=True, silent=True) or {}
    board = payload.get("board")
    ai_player = int(payload.get("player", 2))
    checkpoint_path = payload.get("checkpoint") or app.config.get("CHECKPOINT_PATH")
    if not isinstance(board, list) or len(board) != 6 or not all(isinstance(r, list) and len(r) == 7 for r in board):
        return jsonify({"ok": False, "error": "Board must be a 6x7 list of ints."}), 400

    try:
        model, input_dim = load_model(checkpoint_path)
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "Checkpoint not found. Pass --checkpoint or set the default checkpoint path."}), 404

    try:
        move = choose_action(model, board, input_dim, ai_player)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "column": int(move)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Connect Four policy or play a browser game against it.")
    parser.add_argument("--checkpoint", type=str, required=False, default="checkpoints/policy_net1_ep300.pt", help="Path to a saved checkpoint .pt file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of games to evaluate in CLI mode")
    parser.add_argument("--html", action="store_true", help="Serve the browser-based Connect Four game against the checkpoint")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.html:
        app.config["CHECKPOINT_PATH"] = args.checkpoint
        app.run(host=args.host, port=args.port, debug=False)
    else:
        evaluate(args.checkpoint, args.episodes)
