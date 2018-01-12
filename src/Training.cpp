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
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <boost/utility.hpp>
#include "stdlib.h"
#include "zlib.h"
#include "string.h"

#include "Training.h"
#include "UCTNode.h"
#include "Random.h"
#include "Utils.h"
#include "UCTSearch.h"

std::vector<TimeStep> Training::m_data{};

std::string OutputChunker::gen_chunk_name(void) const {
    auto base = std::string{m_basename};
    base.append("." + std::to_string(m_chunk_count) + ".gz");
    return base;
}

OutputChunker::OutputChunker(const std::string& basename,
                             bool compress)
    : m_basename(basename), m_compress(compress) {
}

OutputChunker::~OutputChunker() {
    flush_chunks();
}

void OutputChunker::append(const std::string& str) {
    m_buffer.append(str);
    m_step_count++;
    if (m_step_count >= CHUNK_SIZE) {
        flush_chunks();
    }
}

void OutputChunker::flush_chunks() {
    if (m_compress) {
        auto chunk_name = gen_chunk_name();
        auto out = gzopen(chunk_name.c_str(), "wb9");

        auto in_buff_size = m_buffer.size();
        auto in_buff = std::make_unique<char[]>(in_buff_size);
        memcpy(in_buff.get(), m_buffer.data(), in_buff_size);

        auto comp_size = gzwrite(out, in_buff.get(), in_buff_size);
        if (!comp_size) {
            throw std::runtime_error("Error in gzip output");
        }
        Utils::myprintf("Writing chunk %d\n",  m_chunk_count);
        gzclose(out);
    } else {
        auto chunk_name = m_basename;
        auto flags = std::ofstream::out | std::ofstream::app;
        auto out = std::ofstream{chunk_name, flags};
        out << m_buffer;
        out.close();
    }

    m_buffer.clear();
    m_chunk_count++;
    m_step_count = 0;
}

void Training::clear_training() {
    Training::m_data.clear();
}

void Training::record(const BoardHistory& state, UCTNode& root) {
    auto step = TimeStep{};
    step.to_move = state.cur().side_to_move();
    step.planes = Network::NNPlanes{};
    Network::gather_features(state, step.planes);

    auto result = Network::get_scored_moves(state);
    step.net_winrate = result.second;

    const auto& best_node = root.get_best_root_child(step.to_move);
    step.root_uct_winrate = root.get_eval(step.to_move);
    step.child_uct_winrate = best_node.get_eval(step.to_move);
    step.bestmove_visits = best_node.get_visits();

    step.probabilities.resize(Network::NUM_OUTPUT_POLICY);

    // Get total visit amount. We count rather
    // than trust the root to avoid ttable issues.
    auto sum_visits = 0.0;
    for (const auto& child : root.get_children()) {
        sum_visits += child->get_visits();
    }

    // In a terminal position, we can have children, but we will not able to
    // accumulate search results on them because every attempt to evaluate will
    // bail immediately. So in this case there will be 0 total visits, and we
    // should not construct the (non-existent) probabilities.
    if (sum_visits <= 0.0) {
        return;
    }

    for (const auto& child : root.get_children()) {
        auto prob = static_cast<float>(child->get_visits() / sum_visits);
        auto move = child->get_move();
        step.probabilities[Network::move_lookup[move]] = prob;
    }

    m_data.emplace_back(step);
}

void Training::dump_training(int game_score, const std::string& out_filename) {
    auto chunker = OutputChunker{out_filename, true};
    dump_training(game_score, chunker);
}

void Training::dump_training(int game_score, OutputChunker& outchunk) {
    for (const auto& step : m_data) {
        std::stringstream out;
        int kFeatureBase = Network::T_HISTORY * 14;
        for (int p = 0; p < kFeatureBase; p++) {
            const auto& plane = step.planes.bit[p];
            // Write it out as a string of hex characters
            for (auto bit = size_t{0}; bit + 3 < plane.size(); bit += 4) {
                auto hexbyte =  plane[bit]     << 3
                              | plane[bit + 1] << 2
                              | plane[bit + 2] << 1
                              | plane[bit + 3] << 0;
                out << std::hex << hexbyte;
            }
            assert(plane.size() % 4 == 0);
            out << std::dec << std::endl;
        }
        for (int i = 0; i < 5; ++i) {
            out << (step.planes.bit[kFeatureBase + i][0] ? "1" : "0") << std::endl;
        }
        out << step.planes.rule50_count << std::endl;
        out << step.planes.move_count << std::endl;
        // Then the move probabilities
        for (auto it = begin(step.probabilities); it != end(step.probabilities); ++it) {
            out << *it;
            if (boost::next(it) != end(step.probabilities)) {
                out << " ";
            }
        }
        out << std::endl;
        // And the game result for the side to move
        out << (step.to_move == BLACK ? -game_score : game_score) << std::endl;
        outchunk.append(out.str());
    }
}

void Training::dump_stats(const std::string& filename) {
    auto chunker = OutputChunker{filename, true};
    dump_stats(chunker);
}

void Training::dump_stats(OutputChunker& outchunk) {
    {
        std::stringstream out;
        out << "1" << std::endl; // File format version 1
        outchunk.append(out.str());
    }
    for (const auto& step : m_data) {
        std::stringstream out;
        out << step.net_winrate
            << " " << step.root_uct_winrate
            << " " << step.child_uct_winrate
            << " " << step.bestmove_visits << std::endl;
        outchunk.append(out.str());
    }
}

//void Training::process_game(Position& state, size_t& train_pos, Color who_won, const std::vector<int>& tree_moves, OutputChunker& outchunker) //--killed

//void Training::dump_supervised(const std::string& sgf_name, const std::string& out_filename) //--killed

