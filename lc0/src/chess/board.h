/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <string>
#include "chess/bitboard.h"
#include "utils/hashcat.h"

namespace lczero {

struct MoveExecution;

// Represents a board position.
// Unlike most chess engines, the board is mirrored for black.
class ChessBoard {
 public:
  static const std::string kStartingFen;

  // Sets position from FEN string.
  // If @no_capture_ply and @moves are not nullptr, they are filled with number
  // of moves without capture and number of full moves since the beginning of
  // the game.
  void SetFromFen(const std::string& fen, int* no_capture_ply = nullptr,
                  int* moves = nullptr);
  // Nullifies the whole structure.
  void Clear();
  // Swaps black and white pieces and mirrors them relative to the
  // middle of the board. (what was on file 1 appears on file 8, what was
  // on rank b remains on b).
  void Mirror();

  // Generates list of possible moves for "ours" (white), but may leave king
  // under check.
  MoveList GeneratePseudolegalMoves() const;
  // Applies the move. (Only for "ours" (white)). Returns true if 50 moves
  // counter should be removed.
  bool ApplyMove(Move move);
  // Checks if the square is under attack from "theirs" (black).
  bool IsUnderAttack(BoardSquare square) const;
  // Checks if "our" (white) king is under check.
  bool IsUnderCheck() const { return IsUnderAttack(our_king_); }
  // Checks whether at least one of the sides has mating material.

  bool HasMatingMaterial() const;
  // Generates legal moves.
  MoveList GenerateLegalMoves() const;
  // Check whether pseudolegal move is legal.
  bool IsLegalMove(Move move, bool was_under_check) const;
  // Returns a list of legal moves and board positions after the move is made.
  std::vector<MoveExecution> GenerateLegalMovesAndPositions() const;

  uint64_t Hash() const {
    return HashCat({our_pieces_.as_int(), their_pieces_.as_int(),
                    rooks_.as_int(), bishops_.as_int(), pawns_.as_int(),
                    our_king_.as_int(), their_king_.as_int(),
                    castlings_.as_int(), flipped_});
  }

  class Castlings {
   public:
    void set_we_can_00() { data_ |= 1; }
    void set_we_can_000() { data_ |= 2; }
    void set_they_can_00() { data_ |= 4; }
    void set_they_can_000() { data_ |= 8; }

    void reset_we_can_00() { data_ &= ~1; }
    void reset_we_can_000() { data_ &= ~2; }
    void reset_they_can_00() { data_ &= ~4; }
    void reset_they_can_000() { data_ &= ~8; }

    bool we_can_00() const { return data_ & 1; }
    bool we_can_000() const { return data_ & 2; }
    bool they_can_00() const { return data_ & 4; }
    bool they_can_000() const { return data_ & 8; }

    void Mirror() { data_ = ((data_ & 0b11) << 2) + ((data_ & 0b1100) >> 2); }

    std::string as_string() const {
      if (data_ == 0) return "-";
      std::string result;
      if (we_can_00()) result += 'K';
      if (we_can_000()) result += 'Q';
      if (they_can_00()) result += 'k';
      if (they_can_000()) result += 'q';
      return result;
    }

    uint8_t as_int() const { return data_; }

    bool operator==(const Castlings& other) const {
      return data_ == other.data_;
    }

   private:
    std::uint8_t data_ = 0;
  };

  std::string DebugString() const;

  BitBoard ours() const { return our_pieces_; }
  BitBoard theirs() const { return their_pieces_; }
  BitBoard pawns() const;
  BitBoard bishops() const { return bishops_ - rooks_; }
  BitBoard rooks() const { return rooks_ - bishops_; }
  BitBoard queens() const { return rooks_ * bishops_; }
  BitBoard our_knights() const {
    return our_pieces_ - pawns() - our_king_ - rooks_ - bishops_;
  }
  BitBoard their_knights() const {
    return their_pieces_ - pawns() - their_king_ - rooks_ - bishops_;
  }
  BitBoard our_king() const { return 1ull << our_king_.as_int(); }
  BitBoard their_king() const { return 1ull << their_king_.as_int(); }
  const Castlings& castlings() const { return castlings_; }
  bool flipped() const { return flipped_; }

  bool operator==(const ChessBoard& other) const {
    return (our_pieces_ == other.our_pieces_) &&
           (their_pieces_ == other.their_pieces_) && (rooks_ == other.rooks_) &&
           (bishops_ == other.bishops_) && (pawns_ == other.pawns_) &&
           (our_king_ == other.our_king_) &&
           (their_king_ == other.their_king_) &&
           (castlings_ == other.castlings_) && (flipped_ == other.flipped_);
  }

  bool operator!=(const ChessBoard& other) const { return !operator==(other); }

 private:
  // All white pieces.
  BitBoard our_pieces_;
  // All black pieces.
  BitBoard their_pieces_;
  // Rooks and queens.
  BitBoard rooks_;
  // Bishops and queens;
  BitBoard bishops_;
  // Pawns.
  // Ranks 1 and 8 have special meaning. Pawn at rank 1 means that
  // corresponding white pawn on rank 4 can be taken en passant. Rank 8 is the
  // same for black pawns. Those "fake" pawns are not present in white_ and
  // black_ bitboards.
  BitBoard pawns_;
  BoardSquare our_king_;
  BoardSquare their_king_;
  Castlings castlings_;
  bool flipped_ = false;  // aka "Black to move".
};

// Stores the move and state of the board after the move is done.
struct MoveExecution {
  Move move;
  ChessBoard board;
  bool reset_50_moves;
};

}  // namespace lczero