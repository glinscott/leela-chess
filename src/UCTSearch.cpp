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

#include "config.h"

#include <assert.h>
#include <limits.h>
#include <cmath>
#include <vector>
#include <utility>
#include <thread>
#include <algorithm>
#include <type_traits>

#include "Position.h"
#include "Movegen.h"
#include "UCI.h"
#include "UCTSearch.h"
#include "Random.h"
#include "Parameters.h"
#include "Utils.h"
#include "Network.h"
#include "Parameters.h"
#include "Training.h"
#include "Types.h"
#include "TimeMan.h"
#ifdef USE_OPENCL
#include "OpenCL.h"
#endif

using namespace Utils;

LimitsType Limits;

UCTSearch::UCTSearch(BoardHistory&& bh)
    : bh_(std::move(bh)) {
    set_playout_limit(cfg_max_playouts);
}

void UCTSearch::set_quiet(bool quiet) {
  quiet_ = quiet;
}

SearchResult UCTSearch::play_simulation(BoardHistory& bh, UCTNode* const node) {
    const auto& cur = bh.cur();
    const auto color = cur.side_to_move();

    auto result = SearchResult{};

    node->virtual_loss();

    if (!node->has_children()) {
        bool drawn = cur.is_draw();
        if (drawn || !MoveList<LEGAL>(cur).size()) {
            float score = (drawn || !cur.checkers()) ? 0.0 : (color == Color::WHITE ? -1.0 : 1.0);
            result = SearchResult::from_score(score);
        } else if (m_nodes < MAX_TREE_SIZE) {
            float eval;
            auto success = node->create_children(m_nodes, bh, eval);
            if (success) {
                result = SearchResult::from_eval(eval);
            }
        }
    }

    if (node->has_children() && !result.valid()) {
        auto next = node->uct_select_child(color);
        auto move = next->get_move();
        bh.do_move(move);
        result = play_simulation(bh, next);
    }

    if (result.valid()) {
        node->update(result.eval());
    }
    node->virtual_loss_undo();

    return result;
}

void UCTSearch::dump_stats(BoardHistory& state, UCTNode& parent) {
    if (cfg_quiet || !parent.has_children()) {
        return;
    }
    myprintf("\n");

    const Color color = state.cur().side_to_move();

    // sort children, put best move on top
    m_root.sort_root_children(color);

    if (parent.get_first_child()->first_visit()) {
        return;
    }

    for (const auto& node : parent.get_children()) {
        std::string tmp = UCI::move(node->get_move());
        std::string pvstring(tmp);

        myprintf("%4s -> %7d (V: %5.2f%%) (N: %5.2f%%) PV: ",
            tmp.c_str(),
            node->get_visits(),
            node->get_eval(color)*100.0f,
            node->get_score() * 100.0f);

        StateInfo si;
        state.cur().do_move(node->get_move(), si);
        pvstring += " " + get_pv(state, *node);
        state.cur().undo_move(node->get_move());

        myprintf("%s\n", pvstring.c_str());
    }
    myprintf("\n");
}

Move UCTSearch::get_best_move() {
    Color color = bh_.cur().side_to_move();

    // Make sure best is first
    m_root.sort_root_children(color);

    // Check whether to randomize the best move proportional
    // to the playout counts.
    if (cfg_randomize) {
        m_root.randomize_first_proportionally();
    }

    Move bestmove = m_root.get_first_child()->get_move();

    // do we have statistics on the moves?
    if (m_root.get_first_child()->first_visit()) {
        return bestmove;
    }

    // should we consider resigning?
    /*
    float bestscore = m_root.get_first_child()->get_eval(color);
    int visits = m_root.get_visits();
    // bad score and visited enough
    if (bestscore < ((float)cfg_resignpct / 100.0f)
        && visits > 500
        && m_rootstate.game_ply() > cfg_min_resign_moves) { //--set cfg_min_resign_moves very high to forbid resigning...?
        myprintf("Score looks bad. Resigning.\n");
        bestmove = MOVE_NONE; //--i guess MOVE_NONE will mean resign.
    }
    */

    return bestmove;
}

std::string UCTSearch::get_pv(BoardHistory& state, UCTNode& parent) {
    if (!parent.has_children()) {
        return std::string();
    }

    auto& best_child = parent.get_best_root_child(state.cur().side_to_move());
    auto best_move = best_child.get_move();
    auto res = UCI::move(best_move);

    StateInfo st;
    state.cur().do_move(best_move, st);

    auto next = get_pv(state, best_child);
    if (!next.empty()) {
        res.append(" ").append(next);
    }
    state.cur().undo_move(best_move);
    return res;
}

void UCTSearch::dump_analysis(int elapsed, bool force_output) {
    if (cfg_quiet && !force_output) {
        return;
    }

    auto bh = bh_.shallow_clone();
    Color color = bh.cur().side_to_move();

    std::string pvstring = get_pv(bh, m_root);
    float feval = m_root.get_eval(color);
    float winrate = 100.0f * feval;
    // UCI-like output wants a depth and a cp.
    // convert winrate to a cp estimate ... assume winrate = 1 / (1 + exp(-cp / 650))
    // (650 can be tuned to have an output more or less matching e.g. SF, once both have similar strength)
    int   cp = -650 * log(1 / feval - 1);
    // same for nodes to depth, assume nodes = 1.8 ^ depth.
    int   depth = log(float(m_nodes)) / log(1.8);
    auto visits = m_root.get_visits();
    printf("info depth %d nodes %d nps %d score cp %d winrate %5.2f%% time %d pv %s\n",
             depth, visits, 1000 * visits / (elapsed + 1),
             cp, winrate, elapsed, pvstring.c_str());
}

bool UCTSearch::is_running() const {
    return m_run && m_nodes < MAX_TREE_SIZE;
}

bool UCTSearch::playout_limit_reached() const {
    return m_playouts >= m_maxplayouts;
}

void UCTWorker::operator()() {
    do {
        BoardHistory bh = bh_.shallow_clone();
        auto result = m_search->play_simulation(bh, m_root);
        if (result.valid()) {
            m_search->increment_playouts();
        }
    } while (m_search->is_running() && !m_search->halt_search());
}

void UCTSearch::increment_playouts() {
    m_playouts++;
}

Move UCTSearch::think() {
    assert(m_playouts == 0);
    assert(m_nodes == 0);

    // set up timing info

    Time.init(bh_.cur().side_to_move(), bh_.cur().game_ply());
    m_target_time = get_search_time();
    m_start_time = now();

    // create a sorted list of legal moves (make sure we play something legal and decent even in time trouble)
    float root_eval;
    m_root.create_children(m_nodes, bh_, root_eval);
    if (cfg_noise) {
        m_root.dirichlet_noise(0.25f, 0.3f);
    }

    m_run = true;
    int cpus = cfg_num_threads;
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cpus; i++) {
        tg.add_task(UCTWorker(bh_, this, &m_root));
    }

    bool keeprunning = true;
    int last_update = 0;
    do {
        auto currstate = bh_.shallow_clone();
        auto result = play_simulation(currstate, &m_root);
        if (result.valid()) {
            increment_playouts();
        }

        // assume nodes = 1.8 ^ depth.
        int depth = log(float(m_nodes)) / log(1.8);
        if (depth != last_update) {
            last_update = depth;
            dump_analysis(Time.elapsed(), false);
        }

        // check if we should still search
        keeprunning = is_running();
        keeprunning &= !halt_search();

    } while(keeprunning);

    // stop the search
    m_run = false;
    tg.wait_all();
    if (!m_root.has_children()) {
        return MOVE_NONE;
    }

    // display search info
    dump_stats(bh_, m_root);
    Training::record(bh_, m_root);

    int milliseconds_elapsed = Time.elapsed();
    if (milliseconds_elapsed > 0) {
        dump_analysis(milliseconds_elapsed, true);
    }
    Move bestmove = get_best_move();
    return bestmove;
}

void UCTSearch::ponder() {
    assert(m_playouts == 0);
    assert(m_nodes == 0);

    m_run = true;
    int cpus = cfg_num_threads;
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cpus; i++) {
        tg.add_task(UCTWorker(bh_, this, &m_root));
    }
    do {
        auto bh = bh_.shallow_clone();
        auto result = play_simulation(bh, &m_root);
        if (result.valid()) {
            increment_playouts();
        }
    } while(!Utils::input_pending() && is_running());

    // stop the search
    m_run = false;
    tg.wait_all();
    // display search info
    myprintf("\n");
    dump_stats(bh_, m_root);

    myprintf("\n%d visits, %d nodes\n\n", m_root.get_visits(), (int)m_nodes);
}

int UCTSearch::get_search_time() {
    if (Limits.use_time_management() && !Limits.dynamic_controls_set()){
        return -1;
    }

    return Limits.movetime ? Limits.movetime : Time.optimum();
}

bool UCTSearch::halt_search() {
    return m_target_time < 0 ? playout_limit_reached() : m_target_time < now() - m_start_time;
}

void UCTSearch::set_playout_limit(int playouts) {
    static_assert(std::is_convertible<decltype(playouts), decltype(m_maxplayouts)>::value, "Inconsistent types for playout amount.");
    if (playouts == 0) {
        m_maxplayouts = std::numeric_limits<decltype(m_maxplayouts)>::max();
    } else {
        m_maxplayouts = playouts;
    }
}



