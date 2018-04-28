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
#include <boost/range/adaptor/reversed.hpp>

#include "Position.h"
#include "Movegen.h"
#include "UCI.h"
#include "UCTSearch.h"
#include "Random.h"
#include "Parameters.h"
#include "Utils.h"
#include "Network.h"
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
    set_visit_limit(cfg_max_visits);
    m_root = std::make_unique<UCTNode>(MOVE_NONE, 0.0f, 0.5f);
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
        auto next = node->uct_select_child(color, node == m_root.get());
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
    m_root->sort_root_children(color);

    if (parent.get_first_child()->first_visit()) {
        return;
    }

    auto root_temperature = 1.0f;
    auto normfactor = float(m_root->get_first_child()->get_visits());
    auto accum = 0.0f;
    if (cfg_randomize) {
        if (cfg_root_temp_decay > 0) {
            root_temperature = get_root_temperature();
        }
        for (const auto& node : boost::adaptors::reverse(parent.get_children())) {
            accum += std::pow(node->get_visits()/normfactor,1/root_temperature);
        }
    }

    // Reverse sort because GUIs typically will reverse it again.
    for (const auto& node : boost::adaptors::reverse(parent.get_children())) {
        std::string tmp = state.cur().move_to_san(node->get_move());
        std::string pvstring(tmp);
        std::string moveprob(10, '\0');

        auto move_probability = 0.0f;
        if (cfg_randomize) {
            move_probability = std::pow(node->get_visits()/normfactor,1/root_temperature)/accum*100.0f;
            if (move_probability > 0.01f) {
                std::snprintf(&moveprob[0], moveprob.size(), "(%6.2f%%)", move_probability);
            } else if (move_probability > 0.00001f) {
                std::snprintf(&moveprob[0], moveprob.size(), "%s", "(> 0.00%)");
            } else {
                std::snprintf(&moveprob[0], moveprob.size(), "%s", "(  0.00%)");
            }
        } else {
            auto needed = std::snprintf(&moveprob[0], moveprob.size(), "%s", " ");
            moveprob.resize(needed+1);
        }
        myprintf_so("info string %5s -> %7d %s (V: %5.2f%%) (N: %5.2f%%) PV: ",
                tmp.c_str(),
                node->get_visits(),
                moveprob.c_str(),
                node->get_eval(color)*100.0f,
                node->get_score() * 100.0f);

        StateInfo si;
        state.cur().do_move(node->get_move(), si);
        // Since this is just a string, set use_san=true
        pvstring += " " + get_pv(state, *node, true);
        state.cur().undo_move(node->get_move());

        myprintf_so("%s\n", pvstring.c_str());
    }
    // winrate separate info string since it's not UCI spec
    float feval = m_root->get_eval(color);
    myprintf_so("info string stm %s winrate %5.2f%%\n",
        color == Color::WHITE ? "White" : "Black", feval * 100.f);
    myprintf("\n");
}

float UCTSearch::get_root_temperature() {
    auto adjusted_ply = 1.0f + (bh_.cur().game_ply()+1.0f) * cfg_root_temp_decay / 50.0f;
    auto root_temp = 1.0f / (1.0f + std::log(adjusted_ply));
    if (root_temp < 0.05f) {
        root_temp = 0.05f;
    }
    return root_temp;
}

Move UCTSearch::get_best_move() {
    Color color = bh_.cur().side_to_move();

    // Make sure best is first
    m_root->sort_root_children(color);

    // Check whether to randomize the best move proportional
    // to the (exponentiated) visit counts.

    if (cfg_randomize) {
        auto root_temperature = 1.0f;
        // If a temperature decay schedule is set, calculate root temperature from
        // ply count and decay constant. Set default value for too small root temperature.
        if (cfg_root_temp_decay > 0) {
            root_temperature = get_root_temperature();
            myprintf("Game ply: %d, root temperature: %5.2f \n",bh_.cur().game_ply()+1, root_temperature);
        }
        m_root->randomize_first_proportionally(root_temperature);
    }

    Move bestmove = m_root->get_first_child()->get_move();

    // do we have statistics on the moves?
    if (m_root->get_first_child()->first_visit()) {
        return bestmove;
    }

    // should we consider resigning?
    /*
       float bestscore = m_root->get_first_child()->get_eval(color);
       int visits = m_root->get_visits();
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

std::string UCTSearch::get_pv(BoardHistory& state, UCTNode& parent, bool use_san) {
    if (!parent.has_children()) {
        return std::string();
    }

    auto& best_child = parent.get_best_root_child(state.cur().side_to_move());
    auto best_move = best_child.get_move();
    auto res = use_san ? state.cur().move_to_san(best_move) : UCI::move(best_move);

    StateInfo st;
    state.cur().do_move(best_move, st);

    auto next = get_pv(state, best_child, use_san);
    if (!next.empty()) {
        res.append(" ").append(next);
    }
    state.cur().undo_move(best_move);
    return res;
}

void UCTSearch::dump_analysis(int64_t elapsed, bool force_output) {
    if (cfg_quiet && !force_output) {
        return;
    }

    auto bh = bh_.shallow_clone();
    Color color = bh.cur().side_to_move();

    // UCI requires long algebraic notation, so use_san=false
    std::string pvstring = get_pv(bh, *m_root, false);
    float feval = m_root->get_raw_eval(color);
    // UCI-like output wants a depth and a cp, so convert winrate to a cp estimate.
    int cp = 290.680623072 * tan(3.096181612 * (feval - 0.5));
    // same for nodes to depth, assume nodes = 1.8 ^ depth.
    int depth = log(float(m_nodes)) / log(1.8);
    // To report nodes, use visits.
    //   - Only includes expanded nodes.
    //   - Includes nodes carried over from tree reuse.
    auto visits = m_root->get_visits();
    // To report nps, use m_playouts to exclude nodes added by tree reuse,
    // which is similar to a ponder hit. The user will expect to know how
    // fast nodes are being added, not how big the ponder hit was.
    myprintf_so("info depth %d nodes %d nps %0.f score cp %d time %lld pv %s\n",
             depth, visits, 1000.0 * m_playouts / (elapsed + 1),
             cp, elapsed, pvstring.c_str());
}

bool UCTSearch::is_running() const {
    return m_run && m_nodes < MAX_TREE_SIZE;
}

int UCTSearch::est_playouts_left() const {
    auto elapsed_millis = now() - m_start_time;
    auto playouts = m_playouts.load();
    if (!Limits.dynamic_controls_set() && !Limits.movetime) {
        // No time control, use playouts or visits.
        const auto playouts_left =
                std::max(0, std::min(m_maxplayouts - playouts,
                                     m_maxvisits - m_root->get_visits()));
        return playouts_left;
    } else if (elapsed_millis < 1000 || playouts < 100) {
        // Until we reach 1 second or 100 playouts playout_rate
        // is not reliable, so just return max.
        return MAXINT_DIV2;
    } else {
        const auto playout_rate = 1.0f * playouts / elapsed_millis;
        const auto time_left = std::max<int>(0, m_target_time - elapsed_millis);
        return static_cast<int>(std::ceil(playout_rate * time_left));
    }
}

size_t UCTSearch::prune_noncontenders() {
    auto Nfirst = 0;
    for (const auto& node : m_root->get_children()) {
        Nfirst = std::max(Nfirst, node->get_visits());
    }
    const auto min_required_visits =
        Nfirst - est_playouts_left();
    auto pruned_nodes = size_t{0};
    for (const auto& node : m_root->get_children()) {
        const auto has_enough_visits =
            node->get_visits() >= min_required_visits;
        node->set_active(has_enough_visits);
        if (!has_enough_visits) {
            ++pruned_nodes;
        }
    }

    return pruned_nodes;
}

bool UCTSearch::have_alternate_moves() {
    if (!cfg_timemanage) {
        // When timemanage is off always return true.
        // Even if there is only one legal move, we need to get
        // an accurate winrate for self play training output.
        return true;
    }
    auto pruned = prune_noncontenders();
    if (pruned == m_root->get_children().size() - 1) {
        auto elapsed_millis = now() - m_start_time;
        if (m_target_time > 0) {
            // TODO: Until we are stable revert to always printing.
            // Later we can put back this term if logging is too spammy.
            //     && m_target_time - elapsed_millis > 500
            // So for now the comment below does not apply.
            // TODO: In a timed search we will essentially always exit because
            // TODO: the remaining time is too short to let another move win, so
            // TODO: avoid spamming this message every move. We'll print it if we
            // TODO: save at least half a second.
            //
            myprintf("Time Budgeted %0.2fs Used %0.2fs Saved %0.2fs (%0.f%%)\n",
                m_target_time / 1000.0f,
                elapsed_millis / 1000.0f,
                (m_target_time - elapsed_millis) / 1000.0f,
                100.0f * (m_target_time - elapsed_millis) / m_target_time);
        }
        return false;
    }
    return true;
}


bool UCTSearch::pv_limit_reached() const {
    return m_playouts >= m_maxplayouts
        || m_root->get_visits() >= m_maxvisits;
}

void UCTWorker::operator()() {
    do {
        BoardHistory bh = bh_.shallow_clone();
        auto result = m_search->play_simulation(bh, m_root);
        if (result.valid()) {
            m_search->increment_playouts();
        }
    } while (m_search->is_running());
}

void UCTSearch::increment_playouts() {
    m_playouts++;
}

Move UCTSearch::think(BoardHistory&& new_bh) {
#ifndef NDEBUG
    auto start_nodes = m_root->count_nodes();
#endif

    uci_stop.store(false, std::memory_order_seq_cst);

    // See if the position is in our previous search tree.
    // If not, construct a new m_root.
    m_root = m_root->find_new_root(m_prevroot_full_key, new_bh);
    if (!m_root) {
        m_root = std::make_unique<UCTNode>(new_bh.cur().get_move(), 0.0f, 0.5f);
    }

    m_playouts = 0;
    m_nodes = m_root->count_nodes();
    // TODO: Both UCI and the next line do shallow_clone.
    // Could optimize this.
    bh_ = new_bh.shallow_clone();
    m_prevroot_full_key = new_bh.cur().full_key();

#ifndef NDEBUG
    myprintf("update_root, %d -> %d expanded nodes (%.1f%% reused)\n",
        start_nodes,
        m_nodes.load(),
        m_nodes > 0 ? 100.0 * m_nodes.load() / start_nodes : 0);
#endif

    // set up timing info

    Time.init(bh_.cur().side_to_move(), bh_.cur().game_ply());
    m_target_time = (Limits.movetime ? Limits.movetime : Time.optimum()) - cfg_lagbuffer_ms;
    m_max_time    = Time.maximum() - cfg_lagbuffer_ms;
    m_start_time  = Limits.timeStarted();

    // create a sorted list of legal moves (make sure we
    // play something legal and decent even in time trouble)
    if (!m_root->has_children()) {
        float root_eval;
        m_root->create_children(m_nodes, bh_, root_eval);
        m_root->update(root_eval);
    }
    if (cfg_noise) {
        m_root->dirichlet_noise(0.25f, 0.3f);
    }

    m_run = true;
    int cpus = cfg_num_threads;
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cpus; i++) {
        tg.add_task(UCTWorker(bh_, this, m_root.get()));
    }

    bool keeprunning = true;
    int last_update = 0;
    do {
        auto currstate = bh_.shallow_clone();
        auto result = play_simulation(currstate, m_root.get());
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
        keeprunning &= !should_halt_search();
        if (!Limits.infinite) {
            // have_alternate_moves has the side effect
            // of pruning moves, so be careful to not even
            // call it when running infinite.
            keeprunning &= have_alternate_moves();
        }
    } while(keeprunning);

    // stop the search
    m_run = false;
    tg.wait_all();
    if (!m_root->has_children()) {
        return MOVE_NONE;
    }

    // reactivate all pruned root children
    for (const auto& node : m_root->get_children()) {
        node->set_active(true);
    }

    // display search info
    dump_stats(bh_, *m_root);
    Training::record(bh_, *m_root);

    int64_t milliseconds_elapsed = now() - m_start_time;
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
        tg.add_task(UCTWorker(bh_, this, m_root.get()));
    }
    do {
        auto bh = bh_.shallow_clone();
        auto result = play_simulation(bh, m_root.get());
        if (result.valid()) {
            increment_playouts();
        }
    } while(!Utils::input_pending() && is_running());

    // stop the search
    m_run = false;
    tg.wait_all();
    // display search info
    myprintf("\n");
    dump_stats(bh_, *m_root);

    myprintf("\n%d visits, %d expanded nodes\n\n", m_root->get_visits(), (int)m_nodes);
}

// Used to check if we've run out of time or reached out playout limit
bool UCTSearch::should_halt_search() {
    if (uci_stop.load(std::memory_order_seq_cst)) return true;
    if (Limits.infinite) return false;
    auto elapsed_millis = now() - m_start_time;
    if (Limits.movetime)
        return (elapsed_millis > m_target_time);
    if (Limits.dynamic_controls_set())
        return (elapsed_millis > m_target_time || elapsed_millis > m_max_time);
    return pv_limit_reached();
}

// Asks the search to stop politely
void UCTSearch::please_stop() {
    uci_stop.store(true, std::memory_order_seq_cst);
}

void UCTSearch::set_playout_limit(int playouts) {
    static_assert(std::is_convertible<decltype(playouts), decltype(m_maxplayouts)>::value, "Inconsistent types for playout amount.");
    if (playouts == 0) {
        // Divide max by 2 to prevent overflow when multithreading.
        m_maxplayouts = MAXINT_DIV2;
    } else {
        m_maxplayouts = playouts;
    }
}

void UCTSearch::set_visit_limit(int visits) {
    static_assert(std::is_convertible<decltype(visits), decltype(m_maxvisits)>::value, "Inconsistent types for visits amount.");
    if (visits == 0) {
        // Divide max by 2 to prevent overflow when multithreading.
        m_maxvisits = MAXINT_DIV2;
    } else {
        m_maxvisits = visits;
    }
}



