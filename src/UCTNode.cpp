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
#ifdef USE_OPENCL
#include "OpenCL.h"
#endif

using namespace Utils;

UCTNode::UCTNode(Move move, float score, float init_eval)
    : m_move(move), m_score(score), m_init_eval(init_eval) {
    assert(m_score >= 0.0 && m_score <= 1.0);
}

UCTNode::~UCTNode() {
    LOCK(get_mutex(), lock);
    UCTNode* next = m_firstchild;

    while (next != nullptr) {
        UCTNode* tmp = next->m_nextsibling;
        delete next;
        next = tmp;
    }
}

bool UCTNode::first_visit() const {
    return m_visits == 0;
}

void UCTNode::link_child(UCTNode* newchild) {
    newchild->m_nextsibling = m_firstchild;
    m_firstchild = newchild;
}

SMP::Mutex & UCTNode::get_mutex() {
    return m_nodemutex;
}

bool UCTNode::create_children(std::atomic<int>& nodecount, Position& state, float& eval) {
    // check whether somebody beat us to it (atomic)
    if (has_children()) {
        return false;
    }
    // acquire the lock
    LOCK(get_mutex(), lock);
    // no successors in final state
    bool drawn = state.is_draw(); //--figure out _ply_ parameter for is_draw...
    if (drawn) {
        return false;
    }
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

    auto raw_netlist = Network::get_scored_moves(&state);
    if (raw_netlist.first.empty()) {
        return false;
    }

    // DCNN returns winrate as side to move
    auto net_eval = raw_netlist.second;
    auto to_move = state.side_to_move();
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

void UCTNode::link_nodelist(std::atomic<int>& nodecount, std::vector<std::pair<float, Move>>& nodelist, float init_eval) {
    if (!nodelist.size()) {
        return;
    }

    // sort (this will reverse scores, but linking is backwards too)
    std::sort(begin(nodelist), end(nodelist));

    // link the nodes together
    auto childrenadded = 0;

    LOCK(get_mutex(), lock);

    for (const auto& node : nodelist) {
        auto vtx = new UCTNode(node.second, node.first, init_eval);
        link_child(vtx);
        childrenadded++;
    }

    nodecount += childrenadded;
    m_has_children = true;
}

float UCTNode::eval_state(Position& state) {
    auto raw_netlist = Network::get_scored_moves(&state);

    // DCNN returns winrate as side to move
    auto net_eval = raw_netlist.second;

    // But we score from white's point of view
    if (state.side_to_move() == BLACK) {
        net_eval = 1.0f - net_eval;
    }

    return net_eval;
}

void UCTNode::dirichlet_noise(float epsilon, float alpha) {
    auto child = m_firstchild;
    auto child_cnt = size_t{0};

    while (child != nullptr) {
        child_cnt++;
        child = child->m_nextsibling;
    }

    auto dirichlet_vector = std::vector<float>{};

    std::gamma_distribution<float> gamma(alpha, 1.0f);
    for (size_t i = 0; i < child_cnt; i++) {
        dirichlet_vector.emplace_back(gamma(Random::get_Rng()));
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

    child = m_firstchild;
    child_cnt = 0;
    while (child != nullptr) {
        auto score = child->get_score();
        auto eta_a = dirichlet_vector[child_cnt++];
        score = score * (1 - epsilon) + epsilon * eta_a;
        child->set_score(score);
        child = child->m_nextsibling;
    }
}

void UCTNode::randomize_first_proportionally() {
    auto accum_vector = std::vector<uint32>{};

    auto child = m_firstchild;
    auto accum = uint32{0};
    while (child != nullptr) {
        accum += child->get_visits();
        accum_vector.emplace_back(accum);
        child = child->m_nextsibling;
    }

    auto pick = Random::get_Rng().randuint32(accum);
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
    child = m_firstchild;
    auto child_cnt = size_t{0};
    while (child != nullptr) {
        // Because of the early out we can't be swapping the first
        // child. Stop at the predecessor, so we can put the nextsibling
        // pointer.
        if (index == child_cnt + 1) {
            // We stopped one early, so we should have a successor
            assert(child->m_nextsibling != nullptr);
            auto old_first = m_firstchild;
            auto old_next = child->m_nextsibling->m_nextsibling;
            // Set up links for the new first node
            m_firstchild = child->m_nextsibling;
            m_firstchild->m_nextsibling = old_first;
            // Point through our nextsibling ptr
            child->m_nextsibling = old_next;
            return;
        }
        child_cnt++;
        child = child->m_nextsibling;
    }
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

    LOCK(get_mutex(), lock);
    UCTNode* child = m_firstchild;

    // Count parentvisits.
    // We do this manually to avoid issues with transpositions.
    int parentvisits = 0;
    while (child != nullptr) {
        parentvisits += child->get_visits();
        child = child->m_nextsibling;
    }
    float numerator = std::sqrt((double)parentvisits);

    child = m_firstchild;

    while (child != nullptr) {
        // get_eval() will automatically set first-play-urgency
        float winrate = child->get_eval(color);
        float psa = child->get_score();
        float denom = 1.0f + child->get_visits();
        float puct = cfg_puct * psa * (numerator / denom);
        float value = winrate + puct;
        assert(value > -1000.0f);

        if (value > best_value) {
            best_value = value;
            best = child;
        }

        child = child->m_nextsibling;
    }

    assert(best != nullptr);

    return best;
}

class NodeComp : public std::binary_function<UCTNode::sortnode_t, UCTNode::sortnode_t, bool> {
public:
    NodeComp() = default;
    // winrate, visits, score, child
    //        0,     1,     2,     3

    bool operator()(const UCTNode::sortnode_t a, const UCTNode::sortnode_t b) {
        // One node has visits, the other does not
        if (!std::get<1>(a) && std::get<1>(b)) {
            return false;
        }

        if (!std::get<1>(b) && std::get<1>(a)) {
            return true;
        }

        // Neither has visits, sort on prior score
        if (!std::get<1>(a) && !std::get<1>(b)) {
            return std::get<2>(a) > std::get<2>(b);
        }

        // Both have visits, but the same amount, prefer winrate
        if (std::get<1>(a) == std::get<1>(b)) {
            return std::get<0>(a) > std::get<0>(b);
        }

        // Both have different visits, prefer greater visits
        return std::get<1>(a) > std::get<1>(b);
    }
};

void UCTNode::sort_root_children(Color color) {
    LOCK(get_mutex(), lock);
    auto tmp = std::vector<sortnode_t>{};

    auto child = m_firstchild;
    while (child != nullptr) {
        auto visits = child->get_visits();
        auto score = child->get_score();
        if (visits) {
            auto winrate = child->get_eval(color);
            tmp.emplace_back(winrate, visits, score, child);
        } else {
            tmp.emplace_back(0.0f, 0, score, child);
        }
        child = child->m_nextsibling;
    }

    // reverse sort, because list reconstruction is backwards
    std::stable_sort(tmp.rbegin(), tmp.rend(), NodeComp());

    m_firstchild = nullptr;

    for (auto& sortnode : tmp) {
        link_child(std::get<3>(sortnode));
    }
}

/**
 * Helper function to get a sortnode_t
 * eval is set to 0 if no visits instead of first-play-urgency
 */
UCTNode::sortnode_t get_sortnode(Color color, UCTNode* child) {
    auto visits = child->get_visits();
    return UCTNode::sortnode_t(
        visits == 0 ? 0.0f : child->get_eval(color),
        visits,
        child->get_score(),
        child);
}

UCTNode* UCTNode::get_best_root_child(Color color) {
    LOCK(get_mutex(), lock);
    assert(m_firstchild != nullptr);

    NodeComp compare;
    auto child = m_firstchild;
    auto best_child = get_sortnode(color, child);
    while (child != nullptr) {
        auto test = get_sortnode(color, child);
        if (compare(test, best_child)) {
            best_child = test;
        }
        child = child->m_nextsibling;
    }
    return std::get<3>(best_child);
}

UCTNode* UCTNode::get_first_child() const {
    return m_firstchild;
}

UCTNode* UCTNode::get_sibling() const {
    return m_nextsibling;
}

// unsafe in SMP, we don't know if people hold pointers to the
// child which they might dereference
void UCTNode::delete_child(UCTNode * del_child) {
    LOCK(get_mutex(), lock);
    assert(del_child != nullptr);

    if (del_child == m_firstchild) {
        m_firstchild = m_firstchild->m_nextsibling;
        delete del_child;
        return;
    } else {
        UCTNode * child = m_firstchild;
        UCTNode * prev  = nullptr;

        do {
            prev  = child;
            child = child->m_nextsibling;

            if (child == del_child) {
                prev->m_nextsibling = child->m_nextsibling;
                delete del_child;
                return;
            }
        } while (child != nullptr);
    }

    assert(false && "Child to delete not found");
}
