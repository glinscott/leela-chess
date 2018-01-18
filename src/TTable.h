/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

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

#ifndef TTABLE_H_INCLUDED
#define TTABLE_H_INCLUDED

#include <vector>

#include "SMP.h"
#include "UCTNode.h"

class TTEntry {
public:
    TTEntry() = default;

    uint64 m_hash{0};
    int m_visits;
    double m_eval_sum;
};

class TTable {
public:
    /*
        return the global TT
    */
    static TTable* get();

    // Clear the TTable (useful when starting new game)
    void clear();

    // Clear a given entry - used to avoid repetitions.
    void clear_entry(uint64_t hash);

    /*
        update corresponding entry
    */
    void update(uint64 hash, const UCTNode* node);

    /*
        sync given node with TT
    */
    void sync(uint64 hash, UCTNode * node);

private:
    TTable(int size = 500000);

    SMP::Mutex mutex_;
    std::vector<TTEntry> m_buckets;
};

#endif
