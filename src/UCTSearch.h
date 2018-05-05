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
#include <unordered_set>

#include "Position.h"
#include "UCTNode.h"
#include "TimeMan.h"
#include "Utils.h"

// SearchResult is in [0,1]
// 0.0 represents Black win
// 0.5 represents draw
// 1.0 represents White win
// Eg. 0.1 would be a high probability of Black winning.
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

    UCTSearch(BoardHistory&& bh);
    Move think(BoardHistory&& bh);
    void set_playout_limit(int playouts);
    void set_node_limit(int nodes);
    void set_analyzing(bool flag);
    void set_quiet(bool flag);
    void ponder();
    bool is_running() const;
    int est_playouts_left() const;
    size_t prune_noncontenders();
    bool have_alternate_moves();
    bool pv_limit_reached() const;
    void increment_playouts();
    bool should_halt_search();
    void please_stop();
    SearchResult play_simulation(BoardHistory& bh, UCTNode* const node, int sdepth);

private:
    void dump_stats(BoardHistory& pos, UCTNode& parent);
    std::string get_pv(BoardHistory& pos, UCTNode& parent, bool use_san);
    void dump_analysis(int64_t elapsed, bool force_output);
    Move get_best_move();
    float get_root_temperature();

    BoardHistory bh_;
    Key m_prevroot_full_key{0};
    std::unique_ptr<UCTNode> m_root;
    std::atomic<int> m_nodes{0};
    std::atomic<int> m_playouts{0};
    std::atomic<int> m_maxdepth{0};
    std::atomic<int> m_tbhits{0};
    int64_t m_target_time{0};
    int64_t m_max_time{0};
    int64_t m_start_time{0};
    std::atomic<bool> m_run{false};
    int m_maxplayouts;
    int m_maxnodes;

    bool quiet_ = true;
    std::atomic<bool> uci_stop{false};

    std::unordered_set<int> m_tbpruned;

    int get_search_time();
};

class UCTWorker {
public:
    UCTWorker(const BoardHistory& bh, UCTSearch* search, UCTNode* root)
      : bh_(bh), m_search(search), m_root(root) {}
    void operator()();
private:
    const BoardHistory& bh_;
    UCTSearch* m_search;
    UCTNode* m_root;
};

/// LimitsType struct stores information sent by GUI about available time to
/// search the current move, maximum depth/time, or if we are in analysis mode.

struct LimitsType {

    LimitsType() { // Init explicitly due to broken value-initialization of non POD in MSVC
        nodes = time[WHITE] = time[BLACK] = inc[WHITE] = inc[BLACK] =
        npmsec = movestogo = depth = movetime = mate = perft = infinite = 0;
        startTime = now();
    }

    int64_t timeStarted() const { return startTime; }

    bool dynamic_controls_set() const {
        return (time[WHITE] | time[BLACK] | inc[WHITE] | inc[BLACK] | npmsec | movestogo) != 0;
    }

    bool use_time_management() const {
        return !(mate | movetime | depth | nodes | perft | infinite);
    }

    std::vector<Move> searchmoves;
    int time[COLOR_NB], inc[COLOR_NB], npmsec, movestogo, depth,
            movetime, mate, perft, infinite;
    int64_t nodes;
    TimePoint startTime;
};

extern LimitsType Limits;

#endif
