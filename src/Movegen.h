/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2017 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MOVEGEN_H_INCLUDED
#define MOVEGEN_H_INCLUDED

#include <algorithm>

#include "Position.h"
#include "Types.h"

enum GenType {
  CAPTURES,
  QUIETS,
  QUIET_CHECKS,
  EVASIONS,
  NON_EVASIONS,
  LEGAL
};

struct ExtMove {
  Move move;
  int value;

  operator Move() const { return move; }
  void operator=(Move m) { move = m; }

  // Inhibit unwanted implicit conversions to Move
  // with an ambiguity that yields to a compile error.
  operator float() const = delete;
};

inline bool operator<(const ExtMove& f, const ExtMove& s) {
  return f.value < s.value;
}

template<CastlingRight Cr, bool Checks>
ExtMove* generate_castling(const Position& pos, ExtMove* moveList, Color us) {

  static const bool KingSide = (Cr == WHITE_OO || Cr == BLACK_OO);

  if (pos.castling_impeded(Cr) || !pos.can_castle(Cr))
      return moveList;

  // After castling, the rook and king final positions are the same in Chess960
  // as they would be in standard chess.
  Square kfrom = pos.square<KING>(us);
  Square rfrom = pos.castling_rook_square(Cr);
  Square kto = relative_square(us, KingSide ? SQ_G1 : SQ_C1);
  Bitboard enemies = pos.pieces(~us);

  assert(!pos.checkers());

  const Direction K = KingSide ? WEST : EAST;

  for (Square s = kto; s != kfrom; s += K)
      if (pos.attackers_to(s) & enemies)
          return moveList;

  // Because we generate only legal castling moves we need to verify that
  // when moving the castling rook we do not discover some hidden checker.
  // For instance an enemy queen in SQ_A1 when castling rook is in SQ_B1.

  Move m = make<CASTLING>(kfrom, rfrom);

  if (Checks && !pos.gives_check(m))
      return moveList;

  *moveList++ = m;
  return moveList;
}

template<Color Us, GenType Type, bool Checks, bool WithCastle = true>
ExtMove* generate_king_moves(const Position& pos, ExtMove* moveList, Bitboard target) {

  if (Type != QUIET_CHECKS && Type != EVASIONS)
  {
      Square ksq = pos.square<KING>(Us);
      Bitboard b = pos.attacks_from<KING>(ksq) & target;
      while (b)
          *moveList++ = make_move(ksq, pop_lsb(&b));
  }

  if (WithCastle && Type != CAPTURES && Type != EVASIONS && pos.can_castle(Us))
  {
      moveList = generate_castling<MakeCastling<Us,  KING_SIDE>::right, Checks>(pos, moveList, Us);
      moveList = generate_castling<MakeCastling<Us, QUEEN_SIDE>::right, Checks>(pos, moveList, Us);
  }

  return moveList;
}

template<GenType Type, Direction D>
ExtMove* make_promotions(ExtMove* moveList, Square to, Square ksq) {

  if (Type != QUIET_CHECKS)
  {
      *moveList++ = make<PROMOTION>(to - D, to, QUEEN);
      *moveList++ = make<PROMOTION>(to - D, to, ROOK);
      *moveList++ = make<PROMOTION>(to - D, to, BISHOP);
      *moveList++ = make<PROMOTION>(to - D, to, KNIGHT);
  }

  // Knight promotion is the only promotion that can give a direct check
  // that's not already included in the queen promotion.
  if (Type == QUIET_CHECKS && (PseudoAttacks[KNIGHT][to] & ksq))
      *moveList++ = make<PROMOTION>(to - D, to, KNIGHT);
  else
      (void)ksq; // Silence a warning under MSVC

  return moveList;
}

template<Color Us, GenType Type, bool Checks>
ExtMove* generate_castling_moves(const Position& pos, ExtMove* moveList) {

  if (Type != CAPTURES && Type != EVASIONS && pos.can_castle(Us))
  {
      moveList = generate_castling<MakeCastling<Us,  KING_SIDE>::right, Checks>(pos, moveList, Us);
      moveList = generate_castling<MakeCastling<Us, QUEEN_SIDE>::right, Checks>(pos, moveList, Us);
  }

  return moveList;
}

template<Color Us, GenType Type>
ExtMove* generate_pawn_moves(const Position& pos, ExtMove* moveList, Bitboard target) {

  // Compute our parametrized parameters at compile time, named according to
  // the point of view of white side.
  const Color     Them     = (Us == WHITE ? BLACK      : WHITE);
  const Bitboard  TRank8BB = (Us == WHITE ? Rank8BB    : Rank1BB);
  const Bitboard  TRank7BB = (Us == WHITE ? Rank7BB    : Rank2BB);
  const Bitboard  TRank3BB = (Us == WHITE ? Rank3BB    : Rank6BB);
  const Direction Up       = (Us == WHITE ? NORTH      : SOUTH);
  const Direction Right    = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
  const Direction Left     = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);

  Bitboard emptySquares;

  Bitboard pawnsOn7    = pos.pieces(Us, PAWN) &  TRank7BB;
  Bitboard pawnsNotOn7 = pos.pieces(Us, PAWN) & ~TRank7BB;

  Bitboard enemies = (Type == EVASIONS ? pos.pieces(Them) & target:
                      Type == CAPTURES ? target : pos.pieces(Them));

  // Single and double pawn pushes, no promotions
  if (Type != CAPTURES)
  {
      emptySquares = (Type == QUIETS || Type == QUIET_CHECKS ? target : ~pos.pieces());

      Bitboard b1 = shift<Up>(pawnsNotOn7)   & emptySquares;
      Bitboard b2 = shift<Up>(b1 & TRank3BB) & emptySquares;

      if (Type == EVASIONS) // Consider only blocking squares
      {
          b1 &= target;
          b2 &= target;
      }

      if (Type == QUIET_CHECKS)
      {
          Square ksq = pos.square<KING>(Them);

          b1 &= pos.attacks_from<PAWN>(ksq, Them);
          b2 &= pos.attacks_from<PAWN>(ksq, Them);

          // Add pawn pushes which give discovered check. This is possible only
          // if the pawn is not on the same file as the enemy king, because we
          // don't generate captures. Note that a possible discovery check
          // promotion has been already generated amongst the captures.
          Bitboard dcCandidates = pos.discovered_check_candidates();
          if (pawnsNotOn7 & dcCandidates)
          {
              Bitboard dc1 = shift<Up>(pawnsNotOn7 & dcCandidates) & emptySquares & ~file_bb(ksq);
              Bitboard dc2 = shift<Up>(dc1 & TRank3BB) & emptySquares;

              b1 |= dc1;
              b2 |= dc2;
          }
      }

      while (b1)
      {
          Square to = pop_lsb(&b1);
          *moveList++ = make_move(to - Up, to);
      }

      while (b2)
      {
          Square to = pop_lsb(&b2);
          *moveList++ = make_move(to - Up - Up, to);
      }
  }

  // Promotions and underpromotions
  if (pawnsOn7 && (Type != EVASIONS || (target & TRank8BB)))
  {
      if (Type == CAPTURES)
          emptySquares = ~pos.pieces();

      if (Type == EVASIONS)
          emptySquares &= target;

      Bitboard b1 = shift<Right>(pawnsOn7) & enemies;
      Bitboard b2 = shift<Left >(pawnsOn7) & enemies;
      Bitboard b3 = shift<Up   >(pawnsOn7) & emptySquares;

      Square ksq = pos.square<KING>(Them);

      while (b1)
          moveList = make_promotions<Type, Right>(moveList, pop_lsb(&b1), ksq);

      while (b2)
          moveList = make_promotions<Type, Left >(moveList, pop_lsb(&b2), ksq);

      while (b3)
          moveList = make_promotions<Type, Up   >(moveList, pop_lsb(&b3), ksq);
  }

  // Standard and en-passant captures
  if (Type == CAPTURES || Type == EVASIONS || Type == NON_EVASIONS)
  {
      Bitboard b1 = shift<Right>(pawnsNotOn7) & enemies;
      Bitboard b2 = shift<Left >(pawnsNotOn7) & enemies;

      while (b1)
      {
          Square to = pop_lsb(&b1);
          *moveList++ = make_move(to - Right, to);
      }

      while (b2)
      {
          Square to = pop_lsb(&b2);
          *moveList++ = make_move(to - Left, to);
      }

      if (pos.ep_square() != SQ_NONE)
      {
          assert(rank_of(pos.ep_square()) == relative_rank(Us, RANK_6));

          // An en passant capture can be an evasion only if the checking piece
          // is the double pushed pawn and so is in the target. Otherwise this
          // is a discovery check and we are forced to do otherwise.
          if (Type == EVASIONS && !(target & (pos.ep_square() - Up)))
              return moveList;

          b1 = pawnsNotOn7 & pos.attacks_from<PAWN>(pos.ep_square(), Them);

          assert(b1);

          while (b1)
              *moveList++ = make<ENPASSANT>(pop_lsb(&b1), pos.ep_square());
      }
  }

  return moveList;
}

template<PieceType Pt, bool Checks>
ExtMove* generate_moves(const Position& pos, ExtMove* moveList, Color us,
                        Bitboard target) {

  assert(Pt != KING && Pt != PAWN);

  const Square* pl = pos.squares<Pt>(us);

  for (Square from = *pl; from != SQ_NONE; from = *++pl)
  {
      if (Checks)
      {
          if (    (Pt == BISHOP || Pt == ROOK || Pt == QUEEN)
              && !(PseudoAttacks[Pt][from] & target & pos.check_squares(Pt)))
              continue;

          if (pos.discovered_check_candidates() & from)
              continue;
      }

      Bitboard b = pos.attacks_from<Pt>(from) & target;

      if (Checks)
          b &= pos.check_squares(Pt);

      while (b)
          *moveList++ = make_move(from, pop_lsb(&b));
  }

  return moveList;
}


template<GenType>
ExtMove* generate(const Position& pos, ExtMove* moveList);

/// The MoveList struct is a simple wrapper around generate(). It sometimes comes
/// in handy to use this class instead of the low level generate() function.
template<GenType T>
struct MoveList {

  explicit MoveList(const Position& pos) : last(generate<T>(pos, moveList)) {}
  const ExtMove* begin() const { return moveList; }
  const ExtMove* end() const { return last; }
  size_t size() const { return last - moveList; }
  bool contains(Move move) const {
    return std::find(begin(), end(), move) != end();
  }

private:
  ExtMove moveList[MAX_MOVES], *last;
};

#endif // #ifndef MOVEGEN_H_INCLUDED
