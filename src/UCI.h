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

#ifndef UCI_H_INCLUDED
#define UCI_H_INCLUDED

#include <map>
#include <string>

#include "Types.h"

class Position;
struct BoardHistory;

namespace UCI {
    
class Option;
/// Custom comparator because UCI options should be case insensitive
struct CaseInsensitiveLess {
    bool operator() (const std::string&, const std::string&) const;
};

/// Our options container is actually a std::map
typedef std::map<std::string, Option, CaseInsensitiveLess> OptionsMap;

/// Option class implements an option as defined by UCI protocol
class Option {
protected:
    typedef void (*OnChange)(const Option&);

public:
    Option(OnChange = nullptr);
    Option(bool v, OnChange = nullptr);
    Option(const char* v, OnChange = nullptr);
    Option(int v, int minv, int maxv, OnChange = nullptr);
    
    Option& operator=(const std::string&);
    void operator<<(const Option&);
    operator int() const;
    operator std::string() const;

protected:
    friend std::ostream& operator<<(std::ostream&, const OptionsMap&);
    
    std::string defaultValue, currentValue, type;
    int min, max;
    size_t idx;
    OnChange on_change;
    bool advertise {true};
};

class SilentOption : public Option {
public:
    SilentOption(OnChange f = nullptr) : Option(f) {
        advertise = false;
    }

    SilentOption(bool v, OnChange f = nullptr) : Option(v, f) {
        advertise = false;
    }

    SilentOption(const char* v, OnChange f = nullptr) : Option(v, f) {
        advertise = false;
    }

    SilentOption(int v, int minv, int maxv, OnChange f = nullptr) : Option(v, minv, maxv, f) {
        advertise = false;
    }
};

void init(OptionsMap&);
void loop(const std::string& start);
std::string square(Square s);
std::string move(Move m);
Move to_move(const Position& pos, std::string const& str);

template<bool Root>
uint64_t perft(BoardHistory& bh, Depth depth);
} // namespace UCI

extern UCI::OptionsMap Options;

#endif // #ifndef UCI_H_INCLUDED

