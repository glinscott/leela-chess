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

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include "Movegen.h"
#include "Position.h"
#include "UCI.h"

using namespace std;

// FEN string of the initial position, normal chess
const char* StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// UCI::square() converts a Square to a string in algebraic notation (g1, a7, etc.)

std::string UCI::square(Square s) {
    return std::string{ char('a' + file_of(s)), char('1' + rank_of(s)) };
}

/// UCI::move() converts a Move to a string in coordinate notation (g1f3, a7a8q).
/// The only special case is castling, where we print in the e1g1 notation in
/// normal chess mode, and in e1h1 notation in chess960 mode. Internally all
/// castling moves are always encoded as 'king captures rook'.

string UCI::move(Move m) {
    
    Square from = from_sq(m);
    Square to = to_sq(m);
    
    if (m == MOVE_NONE)
        return "(none)";
    
    if (m == MOVE_NULL)
        return "0000";
    
    if (type_of(m) == CASTLING)
        to = make_square(to > from ? FILE_G : FILE_C, rank_of(from));
    
    string move = UCI::square(from) + UCI::square(to);
    
    if (type_of(m) == PROMOTION)
        move += " pnbrqk"[promotion_type(m)];
    
    return move;
}


/// UCI::to_move() converts a string representing a move in coordinate notation
/// (g1f3, a7a8q) to the corresponding legal Move, if any.

Move UCI::to_move(const Position& pos, string& str) { 
    for (const auto& m : MoveList<LEGAL>(pos))
        if (str == UCI::move(m))
            return m;
    
    return MOVE_NONE;
}

