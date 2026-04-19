"""AI Chess Agent using LLM to play chess against a human opponent."""

import chess
import chess.svg
import streamlit as st
from anthropic import Anthropic

client = Anthropic()

SYSTEM_PROMPT = """
You are an expert chess player. You will be given the current chess board state in FEN notation
and the move history. Your task is to select the best legal move.

Respond ONLY with a valid move in UCI format (e.g., e2e4, g1f3, e1g1 for castling).
Do not include any explanation or additional text — just the move.
"""


def get_ai_move(board: chess.Board) -> str:
    """Ask the LLM to select the best move given the current board state."""
    legal_moves = [move.uci() for move in board.legal_moves]
    move_history = " ".join([move.uci() for move in board.move_stack])

    user_message = (
        f"Current board (FEN): {board.fen()}\n"
        f"Move history (UCI): {move_history if move_history else 'None'}\n"
        f"Legal moves: {', '.join(legal_moves)}\n"
        f"Select the best move for {'White' if board.turn == chess.WHITE else 'Black'}."
    )

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=16,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text.strip()


def render_board(board: chess.Board) -> str:
    """Render the chess board as an SVG string."""
    return chess.svg.board(board=board, size=400)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "game_over" not in st.session_state:
        st.session_state.game_over = False
    if "status_message" not in st.session_state:
        st.session_state.status_message = "Your turn! You are playing as White."


def main():
    st.set_page_config(page_title="AI Chess Agent", page_icon="♟️", layout="centered")
    st.title("♟️ AI Chess Agent")
    st.caption("Play chess against Claude AI. You are White, AI is Black.")

    init_session_state()

    board: chess.Board = st.session_state.board

    # Display the board
    st.image(render_board(board).encode(), use_container_width=False)

    st.info(st.session_state.status_message)

    if st.session_state.game_over:
        if st.button("New Game"):
            st.session_state.board = chess.Board()
            st.session_state.game_over = False
            st.session_state.status_message = "Your turn! You are playing as White."
            st.rerun()
        return

    # Human move input (White)
    if board.turn == chess.WHITE:
        with st.form("move_form"):
            user_move = st.text_input("Enter your move (UCI format, e.g. e2e4):")
            submitted = st.form_submit_button("Make Move")

        if submitted and user_move:
            try:
                move = chess.Move.from_uci(user_move.strip())
                if move in board.legal_moves:
                    board.push(move)
                    if board.is_game_over():
                        st.session_state.status_message = f"Game over! Result: {board.result()}"
                        st.session_state.game_over = True
                    else:
                        st.session_state.status_message = "AI is thinking..."
                    st.rerun()
                else:
                    st.error("Illegal move. Please try again.")
            except ValueError:
                st.error("Invalid move format. Use UCI notation like e2e4.")

    # AI move (Black)
    elif board.turn == chess.BLACK and not st.session_state.game_over:
        with st.spinner("AI is thinking..."):
            ai_move_uci = get_ai_move(board)
            try:
                ai_move = chess.Move.from_uci(ai_move_uci)
                if ai_move in board.legal_moves:
                    board.push(ai_move)
                    if board.is_game_over():
                        st.session_state.status_message = f"Game over! Result: {board.result()}"
                        st.session_state.game_over = True
                    else:
                        st.session_state.status_message = f"AI played {ai_move_uci}. Your turn!"
                else:
                    st.session_state.status_message = f"AI returned an illegal move ({ai_move_uci}). Please restart."
                    st.session_state.game_over = True
            except ValueError:
                st.session_state.status_message = "AI returned an invalid move. Please restart."
                st.session_state.game_over = True
        st.rerun()


if __name__ == "__main__":
    main()
