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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <limits>
#include <cmath>

#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <random>
#include <numeric>

#include "Position.h"
#include "Parameters.h"
#include "Movegen.h"
#include "UCTNode.h"
#include "UCTSearch.h"
#include "Utils.h"
#include "Network.h"
#include "Random.h"

using namespace Utils;

UCTNode::UCTNode(Move move, float score, float init_eval)
    : m_move(move), m_score(score), m_init_eval(init_eval) {
    assert(m_score >= 0.0 && m_score <= 1.0);
}

UCTNode::~UCTNode() {
    LOCK(m_nodemutex, lock);
    // Empty the children array while the lock is held
    m_children.clear();
}

bool UCTNode::first_visit() const {
    return m_visits == 0;
}

bool UCTNode::create_children(std::atomic<int>& nodecount, const BoardHistory& state, float& eval) {
    // check whether somebody beat us to it (atomic)
    if (has_children()) {
        return false;
    }
    // acquire the lock
    LOCK(m_nodemutex, lock);
    // check whether somebody beat us to it (after taking the lock)
    if (has_children()) {
        return false;
    }
    // Someone else is running the expansion
    if (m_is_expanding) {
        return false;
    }
    // We'll be the one queueing this node for expansion, stop others
    m_is_expanding = true;
    lock.unlock();

    auto raw_netlist = Network::get_scored_moves(state);
    // no successors in final state
    if (raw_netlist.first.empty()) {
        return false;
    }

    // DCNN returns winrate as side to move
    auto net_eval = raw_netlist.second;
    auto to_move = state.cur().side_to_move();
    // our search functions evaluate from white's point of view
    if (to_move == BLACK) {
        net_eval = 1.0f - net_eval;
    }
    eval = net_eval;

    auto legal_sum = 0.0f;
    for (auto m : raw_netlist.first) {
        legal_sum += m.first;
    }

    // If the sum is 0 or a denormal, then don't try to normalize.
    if (legal_sum > std::numeric_limits<float>::min()) {
        // re-normalize after removing illegal moves.
        for (auto& node : raw_netlist.first) {
            node.first /= legal_sum;
        }
    }

    link_nodelist(nodecount, raw_netlist.first, net_eval);

    return true;
}

void UCTNode::link_nodelist(std::atomic<int>& nodecount, std::vector<Network::scored_node>& nodelist, float init_eval) {
    if (nodelist.empty()) {
        return;
    }

    // Use best to worst order, so highest go first
    std::stable_sort(rbegin(nodelist), rend(nodelist));

    LOCK(m_nodemutex, lock);

    m_children.reserve(nodelist.size());
    for (const auto& node : nodelist) {
        m_children.emplace_back(
            std::make_unique<UCTNode>(node.second, node.first, init_eval)
        );
    }

    nodecount += m_children.size();
    m_has_children = true;
}

float UCTNode::eval_state(const BoardHistory& state) {
    auto raw_netlist = Network::get_scored_moves(state);

    // DCNN returns winrate as side to move
    auto net_eval = raw_netlist.second;

    // But we score from white's point of view
    if (state.cur().side_to_move() == BLACK) {
        net_eval = 1.0f - net_eval;
    }

    return net_eval;
}

void UCTNode::dirichlet_noise(float epsilon, float alpha) {
    auto child_cnt = m_children.size();
    auto dirichlet_vector = std::vector<float>{};

    std::gamma_distribution<float> gamma(alpha, 1.0f);
    for (size_t i = 0; i < child_cnt; i++) {
        dirichlet_vector.emplace_back(gamma(Random::GetRng()));
    }

    auto sample_sum = std::accumulate(begin(dirichlet_vector), end(dirichlet_vector), 0.0f);

    // If the noise vector sums to 0 or a denormal, then don't try to
    // normalize.
    if (sample_sum < std::numeric_limits<float>::min()) {
        return;
    }

    for (auto& v: dirichlet_vector) {
        v /= sample_sum;
    }

    child_cnt = 0;
    for (auto& child : m_children) {
        auto score = child->get_score();
        auto eta_a = dirichlet_vector[child_cnt++];
        score = score * (1 - epsilon) + epsilon * eta_a;
        child->set_score(score);
    }
}

void UCTNode::randomize_first_proportionally() {
    auto accum = uint32{0};
    auto accum_vector = std::vector<uint32>{};
    for (const auto& child : m_children) {
        accum += child->get_visits();
        accum_vector.emplace_back(accum);
    }

    auto pick = Random::GetRng().RandInt<std::uint32_t>(accum);
    auto index = size_t{0};
    for (size_t i = 0; i < accum_vector.size(); i++) {
        if (pick < accum_vector[i]) {
            index = i;
            break;
        }
    }

    // Take the early out
    if (index == 0) {
        return;
    }

    // Now swap the child at index with the first child
    assert(index < m_children.size());
    std::iter_swap(begin(m_children), begin(m_children) + index);
}

Move UCTNode::get_move() const {
    return m_move;
}

void UCTNode::virtual_loss() {
    m_virtual_loss += VIRTUAL_LOSS_COUNT;
}

void UCTNode::virtual_loss_undo() {
    m_virtual_loss -= VIRTUAL_LOSS_COUNT;
}

void UCTNode::update(float eval) {
    m_visits++;
    accumulate_eval(eval);
}

bool UCTNode::has_children() const {
    return m_has_children;
}

void UCTNode::set_visits(int visits) {
    m_visits = visits;
}

float UCTNode::get_score() const {
    return m_score;
}

void UCTNode::set_score(float score) {
    m_score = score;
}

int UCTNode::get_visits() const {
    return m_visits;
}

float UCTNode::get_eval(int tomove) const {
    // Due to the use of atomic updates and virtual losses, it is
    // possible for the visit count to change underneath us. Make sure
    // to return a consistent result to the caller by caching the values.
    auto virtual_loss = int{m_virtual_loss};
    auto visits = get_visits() + virtual_loss;
    if (visits > 0) {
        auto whiteeval = get_whiteevals();
        if (tomove == BLACK) {
            whiteeval += static_cast<double>(virtual_loss);
        }
        auto score = static_cast<float>(whiteeval / (double)visits);
        if (tomove == BLACK) {
            score = 1.0f - score;
        }
        return score;
    } else {
        // If a node has not been visited yet, the eval is that of the parent.
        auto eval = m_init_eval;
        if (tomove == BLACK) {
            eval = 1.0f - eval;
        }
        return eval;
    }
}

double UCTNode::get_whiteevals() const {
    return m_whiteevals;
}

void UCTNode::set_whiteevals(double whiteevals) {
    m_whiteevals = whiteevals;
}

void UCTNode::accumulate_eval(float eval) {
    atomic_add(m_whiteevals, (double)eval);
}

UCTNode* UCTNode::uct_select_child(Color color) {
    UCTNode* best = nullptr;
    float best_value = -1000.0f;

    LOCK(m_nodemutex, lock);
    // Count parentvisits.
    // We do this manually to avoid issues with transpositions.
    auto parentvisits = size_t{0};
    for (const auto& child : m_children) {
        parentvisits += child->get_visits();
    }
    float numerator = std::sqrt((double)parentvisits);

    for (const auto& child : m_children) {
        // get_eval() will automatically set first-play-urgency
        auto winrate = child->get_eval(color);
        auto psa = child->get_score();
        auto denom = 1.0f + child->get_visits();
        auto puct = cfg_puct * psa * (numerator / denom);
        auto value = winrate + puct;
        assert(value > -1000.0f);

        if (value > best_value) {
            best_value = value;
            best = child.get();
        }
    }

    assert(best != nullptr);
    return best;
}

class NodeComp : public std::binary_function<UCTNode::node_ptr_t&,
                                             UCTNode::node_ptr_t&, bool> {
public:
    NodeComp(int color) : m_color(color) {};
    bool operator()(const UCTNode::node_ptr_t& a,
                    const UCTNode::node_ptr_t& b) {
        // if visits are not same, sort on visits
        if (a->get_visits() != b->get_visits()) {
            return a->get_visits() < b->get_visits();
        }

        // neither has visits, sort on prior score
        if (a->get_visits() == 0) {
            return a->get_score() < b->get_score();
        }

        // both have same non-zero number of visits
        return a->get_eval(m_color) < b->get_eval(m_color);
    }
private:
    int m_color;
};

void UCTNode::sort_root_children(Color color) {
    LOCK(m_nodemutex, lock);
    std::stable_sort(begin(m_children), end(m_children), NodeComp(color));
    std::reverse(begin(m_children), end(m_children));
}

UCTNode& UCTNode::get_best_root_child(Color color) {
    LOCK(m_nodemutex, lock);
    assert(!m_children.empty());

    return *(std::max_element(begin(m_children), end(m_children),
                              NodeComp(color))->get());
}

UCTNode* UCTNode::get_first_child() const {
    if (m_children.empty()) {
        return nullptr;
    }
    return m_children.front().get();
}

const std::vector<UCTNode::node_ptr_t>& UCTNode::get_children() const {
    return m_children;
}
