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

#include <algorithm>
#include <cassert>
#include <cstddef> // For offsetof()
#include <cstring> // For std::memset, std::memcmp
#include <iomanip>
#include <sstream>

#include "Bitboard.h"
#include "Movegen.h"
#include "Position.h"
#include "Random.h"
#include "UCI.h"

using std::string;

const char* Position::StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
// Use conservative value for now.
// Can tune performance/accuracy tradeoff later.
static constexpr auto RULE50_SCALE = 1;

namespace Zobrist {

  Key psq[PIECE_NB][SQUARE_NB];
  Key enpassant[FILE_NB];
  Key castling[CASTLING_RIGHT_NB];
  Key side;
  Key rule50[102/RULE50_SCALE];
  Key repetitions[3];
}

namespace {

const string PieceToChar(" PNBRQK  pnbrqk");
const string PieceToSAN(" PNBRQK  PNBRQK");

const Piece Pieces[] = { W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
                         B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING };

} // namespace

/// operator<<(Position) returns an ASCII representation of the position
std::ostream& operator<<(std::ostream& os, const Position& pos) {

  os << "\n +---+---+---+---+---+---+---+---+\n";

  for (Rank r = RANK_8; r >= RANK_1; --r)
  {
      for (File f = FILE_A; f <= FILE_H; ++f)
          os << " | " << PieceToChar[pos.piece_on(make_square(f, r))];

      os << " |\n +---+---+---+---+---+---+---+---+\n";
  }

//  os << "\nFen: " << pos.fen() << "\nKey: " << std::hex << std::uppercase
//     << std::setfill('0') << std::setw(16) << pos.key()
//     << std::setfill(' ') << std::dec << "\nCheckers: ";
//
//  for (Bitboard b = pos.checkers(); b; )
//      os << UCI::square(pop_lsb(&b)) << " ";
//
//  if (    int(Tablebases::MaxCardinality) >= popcount(pos.pieces())
//      && !pos.can_castle(ANY_CASTLING))
//  {
//      StateInfo st;
//      Position p;
//      p.set(pos.fen(), pos.is_chess960(), &st, pos.this_thread());
//      Tablebases::ProbeState s1, s2;
//      Tablebases::WDLScore wdl = Tablebases::probe_wdl(p, &s1);
//      int dtz = Tablebases::probe_dtz(p, &s2);
//      os << "\nTablebases WDL: " << std::setw(4) << wdl << " (" << s1 << ")"
//         << "\nTablebases DTZ: " << std::setw(4) << dtz << " (" << s2 << ")";
//  }

  return os;
}


/// Position::init() initializes at startup the various arrays used to compute
/// hash keys.

void Position::init() {

  Random rng(1070372);

  for (Piece pc : Pieces)
      for (Square s = SQ_A1; s <= SQ_H8; ++s)
          Zobrist::psq[pc][s] = rng.RandInt<Key>();

  for (File f = FILE_A; f <= FILE_H; ++f)
      Zobrist::enpassant[f] = rng.RandInt<Key>();

  for (int cr = NO_CASTLING; cr <= ANY_CASTLING; ++cr)
  {
      Zobrist::castling[cr] = 0;
      Bitboard b = cr;
      while (b)
      {
          Key k = Zobrist::castling[1ULL << pop_lsb(&b)];
          Zobrist::castling[cr] ^= k ? k : rng.RandInt<Key>();
      }
  }

  Zobrist::side = rng.RandInt<Key>();
  for (int i = 0; i < 102/RULE50_SCALE; ++i) {
      Zobrist::rule50[i] = rng.RandInt<Key>();
  }
  for (int i = 0; i <= 2; ++i) {
      Zobrist::repetitions[i] = rng.RandInt<Key>();
  }
}

Key Position::full_key() const {
  auto rule50 = std::min(101 / RULE50_SCALE, st->rule50 / RULE50_SCALE);
  auto reps = std::min(2, repetitions_count());
  // NOTE: Network will call this and then repetitions_count
  // on cache misses. Could be optimized.
  return st->key ^ Zobrist::rule50[rule50] ^ Zobrist::repetitions[reps];
}

/// Position::set() initializes the position object with the given FEN string.
/// This function is not very robust - make sure that input FENs are correct,
/// this is assumed to be the responsibility of the GUI.

Position& Position::set(const string& fenStr, StateInfo* si) {
/*
   A FEN string defines a particular position using only the ASCII character set.

   A FEN string contains six fields separated by a space. The fields are:

   1) Piece placement (from white's perspective). Each rank is described, starting
      with rank 8 and ending with rank 1. Within each rank, the contents of each
      square are described from file A through file H. Following the Standard
      Algebraic Notation (SAN), each piece is identified by a single letter taken
      from the standard English names. White pieces are designated using upper-case
      letters ("PNBRQK") whilst Black uses lowercase ("pnbrqk"). Blank squares are
      noted using digits 1 through 8 (the number of blank squares), and "/"
      separates ranks.

   2) Active color. "w" means white moves next, "b" means black.

   3) Castling availability. If neither side can castle, this is "-". Otherwise,
      this has one or more letters: "K" (White can castle kingside), "Q" (White
      can castle queenside), "k" (Black can castle kingside), and/or "q" (Black
      can castle queenside).

   4) En passant target square (in algebraic notation). If there's no en passant
      target square, this is "-". If a pawn has just made a 2-square move, this
      is the position "behind" the pawn. This is recorded only if there is a pawn
      in position to make an en passant capture, and if there really is a pawn
      that might have advanced two squares.

   5) Halfmove clock. This is the number of halfmoves since the last pawn advance
      or capture. This is used to determine if a draw can be claimed under the
      fifty-move rule.

   6) Fullmove number. The number of the full move. It starts at 1, and is
      incremented after Black's move.
*/

  unsigned char col, row, token;
  size_t idx;
  Square sq = SQ_A8;
  std::istringstream ss(fenStr);

  std::memset(this, 0, sizeof(Position));
  std::memset(si, 0, sizeof(StateInfo));
  std::fill_n(&pieceList[0][0], sizeof(pieceList) / sizeof(Square), SQ_NONE);
  st = si;

  ss >> std::noskipws;

  // 1. Piece placement
  while ((ss >> token) && !isspace(token))
  {
      if (isdigit(token))
          sq += (token - '0') * EAST; // Advance the given number of files

      else if (token == '/')
          sq += 2 * SOUTH;

      else if ((idx = PieceToChar.find(token)) != string::npos)
      {
          put_piece(Piece(idx), sq);
          ++sq;
      }
  }

  // 2. Active color
  ss >> token;
  sideToMove = (token == 'w' ? WHITE : BLACK);
  ss >> token;

  // 3. Castling availability. Compatible with 3 standards: Normal FEN standard,
  // Shredder-FEN that uses the letters of the columns on which the rooks began
  // the game instead of KQkq and also X-FEN standard that, in case of Chess960,
  // if an inner rook is associated with the castling right, the castling tag is
  // replaced by the file letter of the involved rook, as for the Shredder-FEN.
  while ((ss >> token) && !isspace(token))
  {
      Square rsq;
      Color c = islower(token) ? BLACK : WHITE;
      Piece rook = make_piece(c, ROOK);

      token = char(toupper(token));

      if (token == 'K')
          for (rsq = relative_square(c, SQ_H1); piece_on(rsq) != rook; --rsq) {}

      else if (token == 'Q')
          for (rsq = relative_square(c, SQ_A1); piece_on(rsq) != rook; ++rsq) {}

      else
          continue;

      set_castling_right(c, rsq);
  }

  // 4. En passant square. Ignore if no pawn capture is possible
  if (   ((ss >> col) && (col >= 'a' && col <= 'h'))
      && ((ss >> row) && (row == '3' || row == '6')))
  {
      st->epSquare = make_square(File(col - 'a'), Rank(row - '1'));

      if (   !(attackers_to(st->epSquare) & pieces(sideToMove, PAWN))
          || !(pieces(~sideToMove, PAWN) & (st->epSquare + pawn_push(~sideToMove))))
          st->epSquare = SQ_NONE;
  }
  else
      st->epSquare = SQ_NONE;

  // 5-6. Halfmove clock and fullmove number
  ss >> std::skipws >> st->rule50 >> gamePly;

  // Convert from fullmove starting from 1 to gamePly starting from 0,
  // handle also common incorrect FEN with fullmove = 0.
  gamePly = std::max(2 * (gamePly - 1), 0) + (sideToMove == BLACK);

  set_state(st);

  assert(pos_is_ok());

  return *this;
}


/// Position::set_castling_right() is a helper function used to set castling
/// rights given the corresponding color and the rook starting square.

void Position::set_castling_right(Color c, Square rfrom) {

  Square kfrom = square<KING>(c);
  CastlingSide cs = kfrom < rfrom ? KING_SIDE : QUEEN_SIDE;
  CastlingRight cr = (c | cs);

  st->castlingRights |= cr;
  castlingRightsMask[kfrom] |= cr;
  castlingRightsMask[rfrom] |= cr;
  castlingRookSquare[cr] = rfrom;

  Square kto = relative_square(c, cs == KING_SIDE ? SQ_G1 : SQ_C1);
  Square rto = relative_square(c, cs == KING_SIDE ? SQ_F1 : SQ_D1);

  for (Square s = std::min(rfrom, rto); s <= std::max(rfrom, rto); ++s)
      if (s != kfrom && s != rfrom)
          castlingPath[cr] |= s;

  for (Square s = std::min(kfrom, kto); s <= std::max(kfrom, kto); ++s)
      if (s != kfrom && s != rfrom)
          castlingPath[cr] |= s;
}


/// Position::set_check_info() sets king attacks to detect if a move gives check

void Position::set_check_info(StateInfo* si) const {

  si->blockersForKing[WHITE] = slider_blockers(pieces(BLACK), square<KING>(WHITE), si->pinnersForKing[WHITE]);
  si->blockersForKing[BLACK] = slider_blockers(pieces(WHITE), square<KING>(BLACK), si->pinnersForKing[BLACK]);

  Square ksq = square<KING>(~sideToMove);

  si->checkSquares[PAWN]   = attacks_from<PAWN>(ksq, ~sideToMove);
  si->checkSquares[KNIGHT] = attacks_from<KNIGHT>(ksq);
  si->checkSquares[BISHOP] = attacks_from<BISHOP>(ksq);
  si->checkSquares[ROOK]   = attacks_from<ROOK>(ksq);
  si->checkSquares[QUEEN]  = si->checkSquares[BISHOP] | si->checkSquares[ROOK];
  si->checkSquares[KING]   = 0;
}


/// Position::set_state() computes the hash keys of the position, and other
/// data that once computed is updated incrementally as moves are made.
/// The function is only used when a new position is set up, and to verify
/// the correctness of the StateInfo data when running in debug mode.

void Position::set_state(StateInfo* si) const {

  si->key = si->materialKey = 0;
  si->checkersBB = attackers_to(square<KING>(sideToMove)) & pieces(~sideToMove);

  set_check_info(si);

  for (Bitboard b = pieces(); b; )
  {
      Square s = pop_lsb(&b);
      Piece pc = piece_on(s);
      si->key ^= Zobrist::psq[pc][s];
  }

  if (si->epSquare != SQ_NONE)
      si->key ^= Zobrist::enpassant[file_of(si->epSquare)];

  if (sideToMove == BLACK)
      si->key ^= Zobrist::side;

  si->key ^= Zobrist::castling[si->castlingRights];
  for (Piece pc : Pieces)
  {
      for (int cnt = 0; cnt < pieceCount[pc]; ++cnt)
          si->materialKey ^= Zobrist::psq[pc][cnt];
  }
}


/// Position::set() is an overload to initialize the position object with
/// the given endgame code string like "KBPKN". It is mainly a helper to
/// get the material key out of an endgame code.

Position& Position::set(const string& code, Color c, StateInfo* si) {

  assert(code.length() > 0 && code.length() < 8);
  assert(code[0] == 'K');

  string sides[] = { code.substr(code.find('K', 1)),      // Weak
                     code.substr(0, code.find('K', 1)) }; // Strong

  std::transform(sides[c].begin(), sides[c].end(), sides[c].begin(), tolower);

  string fenStr = "8/" + sides[0] + char(8 - sides[0].length() + '0') + "/8/8/8/8/"
                       + sides[1] + char(8 - sides[1].length() + '0') + "/8 w - - 0 10";

  return set(fenStr, si);
}


/// Position::fen() returns a FEN representation of the position. In case of
/// Chess960 the Shredder-FEN notation is used. This is mainly a debugging function.

const string Position::fen() const {

  int emptyCnt;
  std::ostringstream ss;

  for (Rank r = RANK_8; r >= RANK_1; --r)
  {
      for (File f = FILE_A; f <= FILE_H; ++f)
      {
          for (emptyCnt = 0; f <= FILE_H && empty(make_square(f, r)); ++f)
              ++emptyCnt;

          if (emptyCnt)
              ss << emptyCnt;

          if (f <= FILE_H)
              ss << PieceToChar[piece_on(make_square(f, r))];
      }

      if (r > RANK_1)
          ss << '/';
  }

  ss << (sideToMove == WHITE ? " w " : " b ");

  if (can_castle(WHITE_OO)) ss << 'K';
  if (can_castle(WHITE_OOO)) ss << 'Q';
  if (can_castle(BLACK_OO)) ss << 'k';
  if (can_castle(BLACK_OOO)) ss <<'q';

  if (!can_castle(WHITE) && !can_castle(BLACK))
      ss << '-';

  ss << (ep_square() == SQ_NONE ? " - " : " " + UCI::square(ep_square()) + " ")
     << st->rule50 << " " << 1 + (gamePly - (sideToMove == BLACK)) / 2;

  return ss.str();
}


/// Position::slider_blockers() returns a bitboard of all the pieces (both colors)
/// that are blocking attacks on the square 's' from 'sliders'. A piece blocks a
/// slider if removing that piece from the board would result in a position where
/// square 's' is attacked. For example, a king-attack blocking piece can be either
/// a pinned or a discovered check piece, according if its color is the opposite
/// or the same of the color of the slider.

Bitboard Position::slider_blockers(Bitboard sliders, Square s, Bitboard& pinners) const {

  Bitboard result = 0;
  pinners = 0;

  // Snipers are sliders that attack 's' when a piece is removed
  Bitboard snipers = (  (PseudoAttacks[  ROOK][s] & pieces(QUEEN, ROOK))
                      | (PseudoAttacks[BISHOP][s] & pieces(QUEEN, BISHOP))) & sliders;

  while (snipers)
  {
    Square sniperSq = pop_lsb(&snipers);
    Bitboard b = between_bb(s, sniperSq) & pieces();

    if (!more_than_one(b))
    {
        result |= b;
        if (b & pieces(color_of(piece_on(s))))
            pinners |= sniperSq;
    }
  }
  return result;
}


/// Position::attackers_to() computes a bitboard of all pieces which attack a
/// given square. Slider attacks use the occupied bitboard to indicate occupancy.

Bitboard Position::attackers_to(Square s, Bitboard occupied) const {

  return  (attacks_from<PAWN>(s, BLACK)    & pieces(WHITE, PAWN))
        | (attacks_from<PAWN>(s, WHITE)    & pieces(BLACK, PAWN))
        | (attacks_from<KNIGHT>(s)         & pieces(KNIGHT))
        | (attacks_bb<  ROOK>(s, occupied) & pieces(  ROOK, QUEEN))
        | (attacks_bb<BISHOP>(s, occupied) & pieces(BISHOP, QUEEN))
        | (attacks_from<KING>(s)           & pieces(KING));
}


/// Position::legal() tests whether a pseudo-legal move is legal

bool Position::legal(Move m) const {

  assert(is_ok(m));

  Color us = sideToMove;
  Square from = from_sq(m);

  assert(color_of(moved_piece(m)) == us);
  assert(piece_on(square<KING>(us)) == make_piece(us, KING));

  // En passant captures are a tricky special case. Because they are rather
  // uncommon, we do it simply by testing whether the king is attacked after
  // the move is made.
  if (type_of(m) == ENPASSANT)
  {
      Square ksq = square<KING>(us);
      Square to = to_sq(m);
      Square capsq = to - pawn_push(us);
      Bitboard occupied = (pieces() ^ from ^ capsq) | to;

      assert(to == ep_square());
      assert(moved_piece(m) == make_piece(us, PAWN));
      assert(piece_on(capsq) == make_piece(~us, PAWN));
      assert(piece_on(to) == NO_PIECE);

      return   !(attacks_bb<  ROOK>(ksq, occupied) & pieces(~us, QUEEN, ROOK))
            && !(attacks_bb<BISHOP>(ksq, occupied) & pieces(~us, QUEEN, BISHOP));
  }

  // If the moving piece is a king, check whether the destination
  // square is attacked by the opponent. Castling moves are checked
  // for legality during move generation.
  if (type_of(piece_on(from)) == KING)
      return type_of(m) == CASTLING || !(attackers_to(to_sq(m)) & pieces(~us));

  // A non-king move is legal if and only if it is not pinned or it
  // is moving along the ray towards or away from the king.
  return   !(pinned_pieces(us) & from)
        ||  aligned(from, to_sq(m), square<KING>(us));
}


/// Position::pseudo_legal() takes a random move and tests whether the move is
/// pseudo legal. It is used to validate moves from TT that can be corrupted
/// due to SMP concurrent access or hash position key aliasing.

bool Position::pseudo_legal(const Move m) const {

  Color us = sideToMove;
  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece pc = moved_piece(m);

  // Use a slower but simpler function for uncommon cases
  if (type_of(m) != NORMAL)
      return MoveList<LEGAL>(*this).contains(m);

  // Is not a promotion, so promotion piece must be empty
  if (promotion_type(m) - KNIGHT != NO_PIECE_TYPE)
      return false;

  // If the 'from' square is not occupied by a piece belonging to the side to
  // move, the move is obviously not legal.
  if (pc == NO_PIECE || color_of(pc) != us)
      return false;

  // The destination square cannot be occupied by a friendly piece
  if (pieces(us) & to)
      return false;

  // Handle the special case of a pawn move
  if (type_of(pc) == PAWN)
  {
      // We have already handled promotion moves, so destination
      // cannot be on the 8th/1st rank.
      if (rank_of(to) == relative_rank(us, RANK_8))
          return false;

      if (   !(attacks_from<PAWN>(from, us) & pieces(~us) & to) // Not a capture
          && !((from + pawn_push(us) == to) && empty(to))       // Not a single push
          && !(   (from + 2 * pawn_push(us) == to)              // Not a double push
               && (rank_of(from) == relative_rank(us, RANK_2))
               && empty(to)
               && empty(to - pawn_push(us))))
          return false;
  }
  else if (!(attacks_from(type_of(pc), from) & to))
      return false;

  // Evasions generator already takes care to avoid some kind of illegal moves
  // and legal() relies on this. We therefore have to take care that the same
  // kind of moves are filtered out here.
  if (checkers())
  {
      if (type_of(pc) != KING)
      {
          // Double check? In this case a king move is required
          if (more_than_one(checkers()))
              return false;

          // Our move must be a blocking evasion or a capture of the checking piece
          if (!((between_bb(lsb(checkers()), square<KING>(us)) | checkers()) & to))
              return false;
      }
      // In case of king moves under check we have to remove king so as to catch
      // invalid moves like b1a1 when opposite queen is on c1.
      else if (attackers_to(to, pieces() ^ from) & pieces(~us))
          return false;
  }

  return true;
}


/// Position::gives_check() tests whether a pseudo-legal move gives a check

bool Position::gives_check(Move m) const {

  assert(is_ok(m));
  assert(color_of(moved_piece(m)) == sideToMove);

  Square from = from_sq(m);
  Square to = to_sq(m);

  // Is there a direct check?
  if (st->checkSquares[type_of(piece_on(from))] & to)
      return true;

  // Is there a discovered check?
  if (   (discovered_check_candidates() & from)
      && !aligned(from, to, square<KING>(~sideToMove)))
      return true;

  switch (type_of(m))
  {
  case NORMAL:
      return false;

  case PROMOTION:
      return attacks_bb(promotion_type(m), to, pieces() ^ from) & square<KING>(~sideToMove);

  // En passant capture with check? We have already handled the case
  // of direct checks and ordinary discovered check, so the only case we
  // need to handle is the unusual case of a discovered check through
  // the captured pawn.
  case ENPASSANT:
  {
      Square capsq = make_square(file_of(to), rank_of(from));
      Bitboard b = (pieces() ^ from ^ capsq) | to;

      return  (attacks_bb<  ROOK>(square<KING>(~sideToMove), b) & pieces(sideToMove, QUEEN, ROOK))
            | (attacks_bb<BISHOP>(square<KING>(~sideToMove), b) & pieces(sideToMove, QUEEN, BISHOP));
  }
  case CASTLING:
  {
      Square kfrom = from;
      Square rfrom = to; // Castling is encoded as 'King captures the rook'
      Square kto = relative_square(sideToMove, rfrom > kfrom ? SQ_G1 : SQ_C1);
      Square rto = relative_square(sideToMove, rfrom > kfrom ? SQ_F1 : SQ_D1);

      return   (PseudoAttacks[ROOK][rto] & square<KING>(~sideToMove))
            && (attacks_bb<ROOK>(rto, (pieces() ^ kfrom ^ rfrom) | rto | kto) & square<KING>(~sideToMove));
  }
  default:
      assert(false);
      return false;
  }
}


/// Position::do_move() makes a move, and saves all information necessary
/// to a StateInfo object. The move is assumed to be legal. Pseudo-legal
/// moves should be filtered out before this function is called.

void Position::do_move(Move m, StateInfo& newSt, bool givesCheck) {

  assert(is_ok(m));
  assert(&newSt != st);

//  thisThread->nodes.fetch_add(1, std::memory_order_relaxed);
  Key k = st->key ^ Zobrist::side;

  // Copy some fields of the old state to our new StateInfo object except the
  // ones which are going to be recalculated from scratch anyway and then switch
  // our state pointer to point to the new (ready to be updated) state.
  std::memcpy(&newSt, st, offsetof(StateInfo, key));
  newSt.previous = st;
  st = &newSt;

  // Increment ply counters. In particular, rule50 will be reset to zero later on
  // in case of a capture or a pawn move.
  ++gamePly;
  ++st->rule50;
  ++st->pliesFromNull;

  Color us = sideToMove;
  Color them = ~us;
  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece pc = piece_on(from);
  Piece captured = type_of(m) == ENPASSANT ? make_piece(them, PAWN) : piece_on(to);

  assert(color_of(pc) == us);
  assert(captured == NO_PIECE || color_of(captured) == (type_of(m) != CASTLING ? them : us));
  assert(type_of(captured) != KING);

  if (type_of(m) == CASTLING)
  {
      assert(pc == make_piece(us, KING));
      assert(captured == make_piece(us, ROOK));

      Square rfrom, rto;
      do_castling<true>(us, from, to, rfrom, rto);

      k ^= Zobrist::psq[captured][rfrom] ^ Zobrist::psq[captured][rto];
      captured = NO_PIECE;
  }

  if (captured)
  {
      Square capsq = to;

      // If the captured piece is a pawn, update pawn hash key, otherwise
      // update non-pawn material.
      if (type_of(captured) == PAWN)
      {
          if (type_of(m) == ENPASSANT)
          {
              capsq -= pawn_push(us);

              assert(pc == make_piece(us, PAWN));
              assert(to == st->epSquare);
              assert(relative_rank(us, to) == RANK_6);
              assert(piece_on(to) == NO_PIECE);
              assert(piece_on(capsq) == make_piece(them, PAWN));

              board[capsq] = NO_PIECE; // Not done by remove_piece()
          }
      }
      // Update board and piece lists
      remove_piece(captured, capsq);

      // Update material hash key and prefetch access to materialTable
      k ^= Zobrist::psq[captured][capsq];
//      prefetch(thisThread->materialTable[st->materialKey]);
      st->materialKey ^= Zobrist::psq[captured][pieceCount[captured]];

      // Reset rule 50 counter
      st->rule50 = 0;
  }

  // Update hash key
  k ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

  // Reset en passant square
  if (st->epSquare != SQ_NONE)
  {
      k ^= Zobrist::enpassant[file_of(st->epSquare)];
      st->epSquare = SQ_NONE;
  }

  // Update castling rights if needed
  if (st->castlingRights && (castlingRightsMask[from] | castlingRightsMask[to]))
  {
      int cr = castlingRightsMask[from] | castlingRightsMask[to];
      k ^= Zobrist::castling[st->castlingRights & cr];
      st->castlingRights &= ~cr;
  }

  // Move the piece. The tricky Chess960 castling is handled earlier
  if (type_of(m) != CASTLING)
      move_piece(pc, from, to);

  // If the moving piece is a pawn do some special extra work
  if (type_of(pc) == PAWN)
  {
      // Set en-passant square if the moved pawn can be captured
      if (   (int(to) ^ int(from)) == 16
          && (attacks_from<PAWN>(to - pawn_push(us), us) & pieces(them, PAWN)))
      {
          st->epSquare = to - pawn_push(us);
          k ^= Zobrist::enpassant[file_of(st->epSquare)];
      }

      else if (type_of(m) == PROMOTION)
      {
          Piece promotion = make_piece(us, promotion_type(m));

          assert(relative_rank(us, to) == RANK_8);
          assert(type_of(promotion) >= KNIGHT && type_of(promotion) <= QUEEN);

          remove_piece(pc, to);
          put_piece(promotion, to);

          // Update hash keys
          k ^= Zobrist::psq[pc][to] ^ Zobrist::psq[promotion][to];
          st->materialKey ^= Zobrist::psq[promotion][pieceCount[promotion] - 1]
              ^ Zobrist::psq[pc][pieceCount[pc]];
      }

      // prefetch access to pawnsTable
//      prefetch2(thisThread->pawnsTable[st->pawnKey]);

      // Reset rule 50 draw counter
      st->rule50 = 0;
  }
  // Set capture piece
  st->capturedPiece = captured;

  // Update the key with the final value
  st->key = k;

  // Calculate checkers bitboard (if move gives check)
  st->checkersBB = givesCheck ? attackers_to(square<KING>(them)) & pieces(us) : 0;

  sideToMove = ~sideToMove;

  // Update king attacks used for fast check detection
  set_check_info(st);
  
  st->move = m;

  assert(pos_is_ok());
}


/// Position::undo_move() unmakes a move. When it returns, the position should
/// be restored to exactly the same state as before the move was made.

void Position::undo_move(Move m) {

  assert(is_ok(m));

  sideToMove = ~sideToMove;

  Color us = sideToMove;
  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece pc = piece_on(to);

  assert(empty(from) || type_of(m) == CASTLING);
  assert(type_of(st->capturedPiece) != KING);

  if (type_of(m) == PROMOTION)
  {
      assert(relative_rank(us, to) == RANK_8);
      assert(type_of(pc) == promotion_type(m));
      assert(type_of(pc) >= KNIGHT && type_of(pc) <= QUEEN);

      remove_piece(pc, to);
      pc = make_piece(us, PAWN);
      put_piece(pc, to);
  }

  if (type_of(m) == CASTLING)
  {
      Square rfrom, rto;
      do_castling<false>(us, from, to, rfrom, rto);
  }
  else
  {
      move_piece(pc, to, from); // Put the piece back at the source square

      if (st->capturedPiece)
      {
          Square capsq = to;

          if (type_of(m) == ENPASSANT)
          {
              capsq -= pawn_push(us);

              assert(type_of(pc) == PAWN);
              assert(to == st->previous->epSquare);
              assert(relative_rank(us, to) == RANK_6);
              assert(piece_on(capsq) == NO_PIECE);
              assert(st->capturedPiece == make_piece(~us, PAWN));
          }

          put_piece(st->capturedPiece, capsq); // Restore the captured piece
      }
  }

  // Finally point our state pointer back to the previous state
  st = st->previous;
  --gamePly;

  assert(pos_is_ok());
}


/// Position::do_castling() is a helper used to do/undo a castling move. This
/// is a bit tricky in Chess960 where from/to squares can overlap.
template<bool Do>
void Position::do_castling(Color us, Square from, Square& to, Square& rfrom, Square& rto) {

  bool kingSide = to > from;
  rfrom = to; // Castling is encoded as "king captures friendly rook"
  rto = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
  to = relative_square(us, kingSide ? SQ_G1 : SQ_C1);

  // Remove both pieces first since squares could overlap in Chess960
  remove_piece(make_piece(us, KING), Do ? from : to);
  remove_piece(make_piece(us, ROOK), Do ? rfrom : rto);
  board[Do ? from : to] = board[Do ? rfrom : rto] = NO_PIECE; // Since remove_piece doesn't do it for us
  put_piece(make_piece(us, KING), Do ? to : from);
  put_piece(make_piece(us, ROOK), Do ? rto : rfrom);
}


/// Position::do(undo)_null_move() is used to do(undo) a "null move": It flips
/// the side to move without executing any move on the board.

void Position::do_null_move(StateInfo& newSt) {

  assert(!checkers());
  assert(&newSt != st);

  std::memcpy(&newSt, st, sizeof(StateInfo));
  newSt.previous = st;
  st = &newSt;

  if (st->epSquare != SQ_NONE)
  {
      st->key ^= Zobrist::enpassant[file_of(st->epSquare)];
      st->epSquare = SQ_NONE;
  }

  st->key ^= Zobrist::side;
//  prefetch(TT.first_entry(st->key));

  ++st->rule50;
  st->pliesFromNull = 0;
  st->move = MOVE_NULL;

  sideToMove = ~sideToMove;

  set_check_info(st);

  assert(pos_is_ok());
}

void Position::undo_null_move() {

  assert(!checkers());

  st = st->previous;
  sideToMove = ~sideToMove;
}

Move Position::get_move() const {
  return st->move;
}


/// Position::key_after() computes the new hash key after the given move. Needed
/// for speculative prefetch. It doesn't recognize special moves like castling,
/// en-passant and promotions.

Key Position::key_after(Move m) const {

  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece pc = piece_on(from);
  Piece captured = piece_on(to);
  Key k = st->key ^ Zobrist::side;

  if (captured)
      k ^= Zobrist::psq[captured][to];

  return k ^ Zobrist::psq[pc][to] ^ Zobrist::psq[pc][from];
}

/// Position::is_draw() tests whether the position is drawn by 50-move rule
/// or by repetition. It does not detect stalemates.

bool Position::is_draw() const {  //--didn't understand this _ply_ parameter; deleting it.

  if (is_draw_by_insufficient_material())
    return true;

  if (st->rule50 > 99 && (!checkers() || MoveList<LEGAL>(*this).size()))
      return true;

  int end = std::min(st->rule50, st->pliesFromNull);

  if (end < 4)
    return false;

  StateInfo* stp = st->previous->previous;
  int cnt = 0;

  for (int i = 4; i <= end; i += 2)
  {
      stp = stp->previous->previous;
      if (stp->key == st->key && ++cnt == 2)
          return true;
  }

  return false;
}

bool Position::is_draw_by_insufficient_material() const {
  switch (count<ALL_PIECES>()) {
  case 2:
    // K v K
    return true;
  case 3:
    // K+B v K or K+N v K
    return pieces(BISHOP, KNIGHT) != 0;
  default:
    // Kings + any number of bishops on the same square color
    return pieces() == pieces(KING, BISHOP) &&
           ((pieces(BISHOP) & DarkSquares) == 0 ||
            (pieces(BISHOP) & ~DarkSquares) == 0);
  }
}

int Position::repetitions_count() const {
  int end = std::min(st->rule50, st->pliesFromNull);
  
  if (end < 4)
    return 0;
  
  StateInfo* stp = st->previous->previous;
  int cnt = 0;
  
  for (int i = 4; i <= end; i += 2)
  {
    stp = stp->previous->previous;
    if (stp->key == st->key)
      cnt++;
  }
  return cnt;
}


// Position::has_repeated() tests whether there has been at least one repetition
// of positions since the last capture or pawn move.

bool Position::has_repeated() const {

    StateInfo* stc = st;
    while (true)
    {
        int i = 4, e = std::min(stc->rule50, stc->pliesFromNull);

        if (e < i)
            return false;

        StateInfo* stp = st->previous->previous;

        do {
            stp = stp->previous->previous;

            if (stp->key == stc->key)
                return true;

            i += 2;
        } while (i <= e);

        stc = stc->previous;
    }
}
/// Position::flip() flips position with the white and black sides reversed. This
/// is only useful for debugging e.g. for finding evaluation symmetry bugs.

void Position::flip() {

  string f, token;
  std::stringstream ss(fen());

  for (Rank r = RANK_8; r >= RANK_1; --r) // Piece placement
  {
      std::getline(ss, token, r > RANK_1 ? '/' : ' ');
      f.insert(0, token + (f.empty() ? " " : "/"));
  }

  ss >> token; // Active color
  f += (token == "w" ? "B " : "W "); // Will be lowercased later

  ss >> token; // Castling availability
  f += token + " ";

  std::transform(f.begin(), f.end(), f.begin(),
                 [](char c) { return char(islower(c) ? toupper(c) : tolower(c)); });

  ss >> token; // En passant square
  f += (token == "-" ? token : token.replace(1, 1, token[1] == '3' ? "6" : "3"));

  std::getline(ss, token); // Half and full moves
  f += token;

  set(f, st);

  assert(pos_is_ok());
}


/// Position::pos_is_ok() performs some consistency checks for the
/// position object and raises an asserts if something wrong is detected.
/// This is meant to be helpful when debugging.

bool Position::pos_is_ok() const {

  const bool Fast = true; // Quick (default) or full check?

  if (   (sideToMove != WHITE && sideToMove != BLACK)
      || piece_on(square<KING>(WHITE)) != W_KING
      || piece_on(square<KING>(BLACK)) != B_KING
      || (   ep_square() != SQ_NONE
          && relative_rank(sideToMove, ep_square()) != RANK_6))
      assert(0 && "pos_is_ok: Default");

  if (Fast)
      return true;

  if (   pieceCount[W_KING] != 1
      || pieceCount[B_KING] != 1
      || attackers_to(square<KING>(~sideToMove)) & pieces(sideToMove))
      assert(0 && "pos_is_ok: Kings");

  if (   (pieces(PAWN) & (Rank1BB | Rank8BB))
      || pieceCount[W_PAWN] > 8
      || pieceCount[B_PAWN] > 8)
      assert(0 && "pos_is_ok: Pawns");

  if (   (pieces(WHITE) & pieces(BLACK))
      || (pieces(WHITE) | pieces(BLACK)) != pieces()
      || popcount(pieces(WHITE)) > 16
      || popcount(pieces(BLACK)) > 16)
      assert(0 && "pos_is_ok: Bitboards");

  for (PieceType p1 = PAWN; p1 <= KING; ++p1)
      for (PieceType p2 = PAWN; p2 <= KING; ++p2)
          if (p1 != p2 && (pieces(p1) & pieces(p2)))
              assert(0 && "pos_is_ok: Bitboards");

  StateInfo si = *st;
  set_state(&si);
  if (std::memcmp(&si, st, sizeof(StateInfo)))
      assert(0 && "pos_is_ok: State");

  for (Piece pc : Pieces)
  {
      if (   pieceCount[pc] != popcount(pieces(color_of(pc), type_of(pc)))
          || pieceCount[pc] != std::count(board, board + SQUARE_NB, pc))
          assert(0 && "pos_is_ok: Pieces");

      for (int i = 0; i < pieceCount[pc]; ++i)
          if (board[pieceList[pc][i]] != pc || index[pieceList[pc][i]] != i)
              assert(0 && "pos_is_ok: Index");
  }

  for (Color c = WHITE; c <= BLACK; ++c)
      for (CastlingSide s = KING_SIDE; s <= QUEEN_SIDE; s = CastlingSide(s + 1))
      {
          if (!can_castle(c | s))
              continue;

          if (   piece_on(castlingRookSquare[c | s]) != make_piece(c, ROOK)
              || castlingRightsMask[castlingRookSquare[c | s]] != (c | s)
              || (castlingRightsMask[square<KING>(c)] & (c | s)) != (c | s))
              assert(0 && "pos_is_ok: Castling");
      }

  return true;
}

std::string Position::move_to_san(Move m) const {
  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece pc = moved_piece(m);

  std::string result = "";
  if (type_of(m) == CASTLING) {
    result = file_of(to) > FILE_E ? "O-O" : "O-O-O";
  } else {
    switch (type_of(pc)) {
    case PAWN: break;
    case KNIGHT: result += "N"; break;
    case BISHOP: result += "B"; break;
    case ROOK: result += "R"; break;
    case QUEEN: result += "Q"; break;
    case KING: result += "K"; break;
    default: break;
    }

    MoveList<LEGAL> moves(*this);
    bool dupe = false, rank_diff = true, file_diff = true;
    for (auto m2 : moves) {
      if (from_sq(m2) != from && to_sq(m2) == to && type_of(pc) == type_of(moved_piece(m2))) {
        dupe = true;
        if (file_of(from) == file_of(from_sq(m2))) file_diff = false;
        if (rank_of(from) == rank_of(from_sq(m2))) rank_diff = false;
      }
    }
    char file = "abcdefgh"[file_of(from)];
    char rank = '1' + rank_of(from);
    if (dupe) {
      if (file_diff) {
        result += file;
      } else if (rank_diff) {
        result += rank;
      } else {
        result += file;
        result += rank;
      }
    } else if (type_of(pc) == PAWN && (board[to] != NO_PIECE || type_of(m) == ENPASSANT)) {
      result += file;
    }

    if (board[to] != NO_PIECE || type_of(m) == ENPASSANT) {
      result += "x";
    }

    result += "abcdefgh"[file_of(to)];
    result += '1' + rank_of(to);
  }
  if (type_of(m) == PROMOTION) {
    switch(promotion_type(m)) {
    case KNIGHT: result += "=N"; break;
    case BISHOP: result += "=B"; break;
    case ROOK: result += "=R"; break;
    case QUEEN: result += "=Q"; break;
    default: break;
    }
  }
  if (gives_check(m)) {
    result += "+";
  }
  return result;
}

/// Position::move_is_san() takes a pseudo-legal Move and a san as input and
/// returns true if moves are equivalent.
template<bool Strict>
bool Position::move_is_san(Move m, const char* ref) const {

  assert(m != MOVE_NONE);

  Bitboard others, b;
  char buf[8], *san = buf;
  Square from = from_sq(m);
  Square to = to_sq(m);
  Piece pc = piece_on(from);
  PieceType pt = type_of(pc);

  buf[2] = '\0'; // Init to fast compare later on

  if (type_of(m) == CASTLING)
  {
      int cmp, last = to > from ? 3 : 5;

      if (ref[0] == 'O')
          cmp = to > from ? strncmp(ref, "O-O", 3) : strncmp(ref, "O-O-O", 5);
      else if (ref[0] == '0')
          cmp = to > from ? strncmp(ref, "0-0", 3) : strncmp(ref, "0-0-0", 5);
      else if (ref[0] == 'o')
          cmp = to > from ? strncmp(ref, "o-o", 3) : strncmp(ref, "o-o-o", 5);
      else
          cmp = 1;

      return !cmp && (ref[last] == '\0' || ref[last] == '+' || ref[last] == '#');
  }

  if (pt != PAWN)
  {
      *san++ = PieceToSAN[pt];

      // A disambiguation occurs if we have more then one piece of type 'pt'
      // that can reach 'to' with a legal move.
      others = b = (attacks_from(type_of(pc), to) & pieces(sideToMove, pt)) ^ from;

      while (Strict && b)
      {
          Square s = pop_lsb(&b);
          if (!legal(make_move(s, to)))
              others ^= s;
      }

      if (!others)
      { /* Disambiguation is not needed */ }

      else if (  !(others & file_bb(from))
               && (Strict || (ref[1] > '8'))) // Check for wrong row disambiguation
          *san++ = char('a' + file_of(from));

      else if (!(others & rank_bb(from)))
          *san++ = char('1' + rank_of(from));

      else
      {
          *san++ = char('a' + file_of(from));
          *san++ = char('1' + rank_of(from));
      }

      if (capture(m) && (Strict || strchr(ref,'x')))
          *san++ =  'x';

      // Add also if not a capture but 'x' is in ref
      else if (!Strict && strchr(ref,'x'))
          *san++ =  'x';
  }
  else if (capture(m))
  {
      *san++ = char('a' + file_of(from));

      if (Strict || strchr(ref,'x'))
          *san++ = 'x';
  }

  *san++ = char('a' + file_of(to));
  *san++ = char('1' + rank_of(to));

  if (type_of(m) == PROMOTION)
  {
      if (Strict) // Sometime promotion move misses the '='
          *san++ = '=';

      *san++ = PieceToSAN[promotion_type(m)];
  }

  if (   buf[1] != ref[1]
      || buf[0] != ref[0])
      return false;

  if (san - buf > 2 && buf[2] != ref[2])
      return false;

  if (!ref[2] || ! ref[3]) // Quiet move both pawn and piece: e4, Nf3
      return true;

  // Be forgivng if the move is missing check annotation
  return !strncmp(ref+3, buf+3, san - buf - 3);
}

// Reduce target to destination square only. It is harmless for castling
// moves because generate_castling() does not use target.

static inline Bitboard trim(Bitboard target, const char* san) {

  if (!san[3] || san[3] == '+')
      return target & make_square(File(san[1] - 'a'), Rank(san[2] - '1'));
  return target;
}

static inline Bitboard trimPawn(Bitboard target, const char* san, bool isCapture) {

  if (isCapture)
  {
      if (san[1] == 'x')
          return target & make_square(File(san[2] - 'a'), Rank(san[3] - '1'));
      else
          // Wrong notation, possibly a uci move like d4xf6, in this case retrun
          // empty target becuase strict search will not find it anyhow
          return 0;
  }
  else
      return target & file_bb(File(san[0] - 'a'));

  return target;
}

Move Position::san_to_move(const std::string& s) const {
  const char* cur = &s[0];
  ExtMove moveList[MAX_MOVES];
  ExtMove* last;
  Color us = sideToMove;

  bool isCapture = strchr(cur, 'x');
  Bitboard target = isCapture ? pieces(~us) : ~pieces();

  switch (cur[0]) {
  case 'N':
      last = generate_moves<KNIGHT, false>(*this, moveList, us, trim(target, cur));
      break;

  case 'B':
      last = generate_moves<BISHOP, false>(*this, moveList, us, trim(target, cur));
      break;

  case 'R':
      last = generate_moves<ROOK , false>(*this, moveList, us, trim(target, cur));
      break;

  case 'Q':
      last = generate_moves<QUEEN, false>(*this, moveList, us, trim(target, cur));
      break;

  case 'K':
        last = us == WHITE ? generate_king_moves<WHITE, NON_EVASIONS, false, false>(*this, moveList, trim(target, cur))
                           : generate_king_moves<BLACK, NON_EVASIONS, false, false>(*this, moveList, trim(target, cur));
      break;

  case 'O':
  case '0':
  case 'o':
      last = us == WHITE ? generate_castling_moves<WHITE, NON_EVASIONS, false>(*this, moveList)
                         : generate_castling_moves<BLACK, NON_EVASIONS, false>(*this, moveList);
      break;

  case '-':
      assert(!strcmp(cur, "--"));
      return MOVE_NULL;

  default:
      assert(cur[0] >= 'a' && cur[0] <= 'h');

      target = trimPawn(target, cur, isCapture);

      if (isCapture)
          last = us == WHITE ? generate_pawn_moves<WHITE, CAPTURES>(*this, moveList, target)
                             : generate_pawn_moves<BLACK, CAPTURES>(*this, moveList, target);
      else
          last = us == WHITE ? generate_pawn_moves<WHITE,   QUIETS>(*this, moveList, target)
                             : generate_pawn_moves<BLACK,   QUIETS>(*this, moveList, target);
      break;
  }

  for (ExtMove* m = moveList; m < last; ++m)
      if (move_is_san(m->move, cur) && legal(m->move))
          return m->move;

  /*
  static bool strict = false;

  if (strict)
      return MOVE_NONE;

  // Retry with disambiguation rule relaxed, this is slow path anyhow
  for (ExtMove* m = moveList; m < last; ++m)
      if (move_is_san<false>(m->move, cur) && legal(m->move))
          return m->move;

  // If is a capture withouth 'x' or a non-capture with 'x' we may have missed
  // it, so regenerate move list to include all legal moves and retry.
  for (const ExtMove& m : MoveList<LEGAL>(*this))
      if (move_is_san<false>(m.move, cur) || move_is_uci(m.move, cur))
          return m.move;

  // Ok, still not fixed, let's try to deduce the move out of the context. Play
  // the game with the generated moves and check if only one candidate is valid.
  //
  // First step is to compute for each move how many plies we can play before
  // a wrong move occurs. This should be done in strict mode to avoid complex
  // artifacts.
  typedef std::pair<Move, const char*> C;
  std::vector<C> candidates;

  strict = true;
  for (ExtMove* m = moveList; m < last; ++m)
      if (legal(m->move))
          candidates.push_back(C{m->move, play_game(*this, m->move, cur, end)});
  strict = false;

  // Then we pick the move that survived the longest
  auto it = std::max_element(candidates.begin(), candidates.end(),
                            [cur](const C& a, const C& b) -> bool
                            {
                                return a.second - cur < b.second - cur;
                            });

  // If the best move is correct until the end we have finished, otherwise
  // replay the game with relaxed checks.
  if (    candidates.size()
      && (it->second == end || play_game(*this, it->first, cur, end) == end))
      return it->first;
  */

  return MOVE_NONE;
}

void BoardHistory::set(const std::string& fen) {
  positions.clear();
  states.clear();

  positions.emplace_back();
  states.emplace_back(new StateInfo());
  cur().set(fen, states.back().get());
}

// Only need to copy the 8 most recent positions, as that's what is needed by
// the eval.  We don't need to fixup StateInfo, as we will never undo_move()
// before the "root" state.
BoardHistory BoardHistory::shallow_clone() const {
  BoardHistory h;
  for (int i = std::max(0, static_cast<int>(positions.size()) - 8); i < static_cast<int>(positions.size()); ++i) {
    h.positions.push_back(positions[i]);
  }
  return h;
}

void BoardHistory::do_move(Move m) {
  states.emplace_back(new StateInfo);
  positions.push_back(positions.back());
  positions.back().do_move(m, *states.back());
}

bool BoardHistory::undo_move() {
	if (positions.size() == 1) return false;
	states.pop_back();
	positions.pop_back();
	return true;
}

std::string BoardHistory::pgn() const {
  std::string result;
  for (int i = 0; i< static_cast<int>(positions.size()) - 1; ++i) {
    if (i % 2 == 0) {
      result += std::to_string(i / 2 + 1) + ". ";
    }
    result += positions[i].move_to_san(positions[i + 1].get_move()) + " ";
  }
  int len = 0;
  int last_space = -1;
  for (int i = 0; i < static_cast<int>(result.size()); ++i) {
    if (result[i] == ' ') {
      last_space = i;
    }
    if (++len >= 76) {
      int remaining = i - last_space;
      result[last_space] = '\n';
      len = remaining;
    }
  }
  return result;
}
