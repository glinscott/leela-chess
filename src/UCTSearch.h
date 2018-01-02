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

#ifndef UCTSEARCH_H_INCLUDED
#define UCTSEARCH_H_INCLUDED

#include <memory>
#include <atomic>
#include <tuple>
#include <unordered_map>

#include "Position.h"
#include "UCTNode.h"

class SearchResult {
public:
    SearchResult() = default;
    bool valid() const { return m_valid;  }
    float eval() const { return m_eval;  }
    static SearchResult from_eval(float eval) {
        return SearchResult(eval);
    }
    static SearchResult from_score(float board_score) {
        if (board_score > 0.0f) {
            return SearchResult(1.0f);
        } else if (board_score < 0.0f) {
            return SearchResult(0.0f);
        } else {
            return SearchResult(0.5f);
        }
    }
private:
    explicit SearchResult(float eval)
        : m_valid(true), m_eval(eval) {}
    bool m_valid{false};
    float m_eval{0.0f};
};

class UCTSearch {
public:
    /*
        Maximum size of the tree in memory. Nodes are about
        40 bytes, so limit to ~1.6G.
    */
    static constexpr auto MAX_TREE_SIZE = 40'000'000;

    UCTSearch(Position& pos, StateListPtr& states);
    Move think();
    void set_playout_limit(int playouts);
    void set_analyzing(bool flag);
    void set_quiet(bool flag);
    void ponder();
    bool is_running() const;
    bool playout_limit_reached() const;
    void increment_playouts();
    SearchResult play_simulation(Position& currstate, UCTNode* const node);
    
private:
    void dump_stats(Position& pos, UCTNode& parent);
    std::string get_pv(Position& pos, UCTNode& parent);
    void dump_analysis(int playouts);
    Move get_best_move();

    Position& m_rootstate;
    StateListPtr& m_statelist;
    UCTNode m_root{MOVE_NONE, 0.0f, 0.5f};
    std::atomic<int> m_nodes{0};
    std::atomic<int> m_playouts{0};
    std::atomic<bool> m_run{false};
    int m_maxplayouts;
};

class UCTWorker {
public:
    UCTWorker(const Position& state, StateListPtr& states, UCTSearch* search, UCTNode* root)
      : m_rootstate(state), m_statelist(new std::deque<StateInfo>(*states)), m_search(search), m_root(root) {}
    void operator()();
private:
    const Position& m_rootstate;
    StateListPtr m_statelist;
    UCTSearch* m_search;
    UCTNode* m_root;
};

#endif
