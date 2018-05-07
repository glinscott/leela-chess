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

#ifndef UCTNODE_H_INCLUDED
#define UCTNODE_H_INCLUDED

#include "config.h"

#include <atomic>
#include <limits>
#include <mutex>
#include <tuple>

#include "Network.h"
#include "Position.h"
#include "SMP.h"
#include <unordered_set>

class UCTNode {
public:
    // When we visit a node, add this amount of virtual losses
    // to it to encourage other CPUs to explore other parts of the
    // search tree.
    static constexpr auto VIRTUAL_LOSS_COUNT = 3;

    using node_ptr_t = std::unique_ptr<UCTNode>;

    explicit UCTNode(Move move, float score, float init_eval);
    UCTNode() = delete;
    ~UCTNode();
    size_t count_nodes() const;
    bool first_visit() const;
    bool has_children() const;
    void set_active(const bool active);
    bool active() const;
    bool create_children(std::atomic<int> & nodecount, const BoardHistory& state, float& eval);
    Move get_move() const;
    int get_visits() const;
    float get_score() const;
    void set_score(float score);
    float get_eval(int tomove) const;
    float get_raw_eval(int tomove) const;
    double get_whiteevals() const;
    void set_visits(int visits);
    void set_whiteevals(double whiteevals);
    void accumulate_eval(float eval);
    void virtual_loss(void);
    void virtual_loss_undo(void);
    void dirichlet_noise(float epsilon, float alpha);
    std::vector<float> calc_proportional(float tau, Color color);
    void randomize_first_proportionally(float tau, Color color);
    void ensure_first_not_pruned(const std::unordered_set<int>& pruned_moves);
    void update(float eval = std::numeric_limits<float>::quiet_NaN());

    UCTNode* uct_select_child(Color color, bool is_root);
    UCTNode* get_first_child() const;
    const std::vector<node_ptr_t>& get_children() const;

    void sort_root_children(Color color);
    UCTNode& get_best_root_child(Color color);
    UCTNode::node_ptr_t find_new_root(Key prevroot_full_key, BoardHistory& new_bh);
    UCTNode::node_ptr_t find_path(std::vector<Move>& moves);

private:
    enum Status : char {
        //INVALID, // superko for Go, NA for chess.
        PRUNED,
        ACTIVE
    };
    void link_nodelist(std::atomic<int>& nodecount, std::vector<Network::scored_node>& nodelist, float init_eval);

    // Move
    Move m_move;
    // UCT
    std::atomic<int16_t> m_virtual_loss{0};
    std::atomic<int> m_visits{0};
    // UCT eval
    float m_score;
    float m_init_eval;
    std::atomic<double> m_whiteevals{0};
    std::atomic<Status> m_status{ACTIVE};
    // Is someone adding scores to this node?
    // We don't need to unset this.
    bool m_is_expanding{false};
    SMP::Mutex m_nodemutex;

    // Tree data
    std::atomic<bool> m_has_children{false};
    std::vector<node_ptr_t> m_children;
};

#endif
