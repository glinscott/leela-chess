/*
    This file is part of Leela Zero.
    Copyright (C) 2018 Gian-Carlo Pascutto

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef LEELA_CHESS_TIMEMAN_H
#define LEELA_CHESS_TIMEMAN_H

#include "Utils.h"
#include "Types.h"
#include "UCTSearch.h"

/// The TimeManagement class computes the optimal time to think depending on
/// the maximum available time, the game move number and other parameters.

class TimeManagement {
public:
    void init(Color us, int ply);
    int optimum() const { return optimumTime; }
    int maximum() const { return maximumTime; }
    int elapsed() const { return int(now() - startTime); }

private:
    TimePoint startTime;
    int optimumTime;
    int maximumTime;
};

extern TimeManagement Time;


#endif //LEELA_CHESS_TIMEMAN_H
