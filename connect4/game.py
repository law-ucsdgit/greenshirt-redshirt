from __future__ import annotations

from typing import Tuple

import numpy as np

BOARD_ROWS = 6
BOARD_COLS = 7
ACTION_SPACE = list(range(BOARD_COLS))

state = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
player1_mask = 0b0
player2_mask = 0b0
currentPlayer = 0


def _bit_for_position(action: int, row: int) -> int:
    return 1 << (action * 7 + row)


def check_win() -> bool:
    """Checks if the player who just moved won the game using bitwise shifts."""
    global currentPlayer, player1_mask, player2_mask
    mask = player1_mask if currentPlayer == 0 else player2_mask
    directions = [7, 1, 6, 8]

    for d in directions:
        step1 = mask & (mask >> d)
        if (step1 & (step1 >> (2 * d))) != 0:
            return True
    return False


def updateState(action: int):
    """Drop a token into the chosen column.

    Returns: (could_update, blocked, next_state)
    """
    global state, player1_mask, player2_mask, currentPlayer

    blocked = False
    for row_idx in range(BOARD_ROWS - 1, -1, -1):
        if state[row_idx, action] == 0:
            state[row_idx, action] = -1 if currentPlayer == 0 else 1
            move_bit = _bit_for_position(action, row_idx)
            if currentPlayer == 0:
                player1_mask |= move_bit
            else:
                player2_mask |= move_bit

            previous_player = currentPlayer
            # Simulate the opponent's move to see whether the bot blocked a win.
            if currentPlayer == 0:
                player2_mask_backup = player2_mask
                player2_mask = 0
                if check_win():
                    blocked = True
                player2_mask = player2_mask_backup
            else:
                player1_mask_backup = player1_mask
                player1_mask = 0
                if check_win():
                    blocked = True
                player1_mask = player1_mask_backup

            currentPlayer = 1 - currentPlayer
            return True, blocked, state.copy()

    return False, blocked, state.copy()


class rlEnvironment:
    def __init__(self):
        self.reset()
        self.action_space = ACTION_SPACE
        self.steps = 0

    def reset(self):
        global state, currentPlayer, player1_mask, player2_mask
        state[:] = 0
        currentPlayer = 0
        player1_mask = 0b0
        player2_mask = 0b0
        self.steps = 0
        return state.copy()

    def step(self, action: int):
        global state, currentPlayer
        could_update, self.blocked, ns = updateState(action)
        reward = 0.0

        if not could_update:
            reward = -100.0
            done = True
            return ns, reward, done, False, {}

        done = check_win()
        if self.blocked:
            reward += 3.0
        if done:
            reward += 100.0
        if action == 3:
            reward += 0.1

        self.steps += 1
        return ns, reward, done, False, {}


Connect4Env = rlEnvironment
