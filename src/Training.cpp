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
#include <boost/filesystem.hpp>
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
                             bool compress,
                             size_t num_games)
    : m_basename(basename), m_compress(compress), m_games_per_chunk(num_games) {
    namespace fs = boost::filesystem;
    m_chunk_count = std::count_if(
        fs::directory_iterator(fs::path(basename).parent_path()),
        fs::directory_iterator(),
        static_cast<bool(*)(const fs::path&)>(fs::is_regular_file));
    Utils::myprintf("Found %d existing chunks in %s\n", m_chunk_count, basename.c_str());
}

OutputChunker::~OutputChunker() {
    flush_chunk();
}

void OutputChunker::append(const std::string& str) {
    m_buffer.append(str);
    m_game_count++;
    if (m_game_count >= m_games_per_chunk) {
        flush_chunk();
    }
}

void OutputChunker::flush_chunk() {
    if (m_buffer.empty()) {
        return;
    }

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
    m_game_count = 0;
}

void Training::clear_training() {
    Training::m_data.clear();
}

// Used by supervised learning
void Training::record(const BoardHistory& state, Move move) {
    auto step = TimeStep{};
    step.to_move = state.cur().side_to_move();
    step.planes = Network::NNPlanes{};
    Network::gather_features(state, step.planes);

    // TODO: Does the SL flow require you to load a network file?
    // Because now Network parses the file and stores m_format_version.
    // Probably we will need a setter function
    // e.g. Network::set_format_version(2)
    throw std::runtime_error("Need to update SL flow");
    step.probabilities.resize(Network::get_num_output_policy());
    step.probabilities[Network::lookup(move, state.cur().side_to_move())] = 1.0;
    m_data.emplace_back(step);
}

// Used by self play
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

    step.probabilities.resize(Network::get_num_output_policy());

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
        step.probabilities[Network::lookup(move, state.cur().side_to_move())] = prob;
    }

    m_data.emplace_back(step);
}

void Training::dump_training(int game_score, const std::string& out_filename) {
    auto chunker = OutputChunker{out_filename, true};
    dump_training(game_score, chunker);
}

// Reverse bit order.
// Required for all dump_training that use binary format.
// (VERSION2 and up)
Network::BoardPlane fix_v2(Network::BoardPlane plane) {
    for (int i = 0, n = plane.size(); i < n; i+=8) {
        for (auto j = 0; j < 4; j++) {
            bool t = plane[i+j];
            plane[i+j] = plane[i+8-j-1];
            plane[i+8-j-1] = t;
        }
    }

    return plane;
}


void Training::dump_training_v2(int game_score, OutputChunker& outchunk) {
    // See chunkparser.py for exact format.
    // format_version 1 was for VERSION1 and VERSION2 of traininig data.
    // format_version 2 must used VERSION3 of training data.
    static int VERSION = Network::get_format_version() + 1;
    assert(VERSION == 2 || VERSION == 3);

    std::stringstream out;
    for (const auto& step : m_data) {
        // Store the binary version number (4 bytes)
        out.write(reinterpret_cast<char*>(&VERSION), sizeof(VERSION));

        // Then the move probabilities
        assert(step.probabilities.size() == Network::get_num_output_policy());
        for (auto p : step.probabilities) {
            uint32 *vp = reinterpret_cast<uint32*>(&p);
            uint32 v = htole32(*vp);
            out.write(reinterpret_cast<char*>(&v), sizeof(v));
        }

        // bitplanes
        int kFeatureBase = Network::T_HISTORY * Network::get_hist_planes();
        for (int p = 0; p < kFeatureBase; p++) {
            const auto& plane = fix_v2(step.planes.bit[p]);
            auto val = htole64(plane.to_ullong());
            assert(plane.size() == 64);
            out.write(reinterpret_cast<char*>(&val), sizeof(val));
        }

        // castling and side to move (5 bytes)
        for (int i = 0; i < 5; ++i) {
            auto bit = static_cast<std::uint8_t>(step.planes.bit[kFeatureBase+i][0]);
            out.write(reinterpret_cast<char*>(&bit), 1);
        }

        // rule 50 (1 byte)
        auto rule50 = static_cast<std::uint8_t>(std::min(255, step.planes.rule50_count));
        out.write(reinterpret_cast<char*>(&rule50), 1);

        // move count (1 byte)
        auto move_count = static_cast<std::uint8_t>(std::min(255, step.planes.move_count));
        out.write(reinterpret_cast<char*>(&move_count), 1);

        // And the game result (1 byte)
        auto result = static_cast<std::int8_t>(step.to_move == BLACK ? -game_score : game_score);
        out.write(reinterpret_cast<char*>(&result), 1);
    }
    assert(Network::get_format_version() == 1
        ? out.str().size() == m_data.size() * 8604
        : out.str().size() == m_data.size() * 8276);
    outchunk.append(out.str());
}

void Training::dump_training(int game_score, OutputChunker& outchunk) {
    std::stringstream out;
    for (const auto& step : m_data) {
        int kFeatureBase = Network::T_HISTORY * 14;
        for (int p = 0; p < kFeatureBase; p++) {
            const auto& plane = step.planes.bit[p];
            // Write it out as a string of hex characters
            for (auto bit = size_t{0}; bit + 3 < plane.size(); bit += 4) {
                auto hexdigit =  plane[bit]     << 3
                               | plane[bit + 1] << 2
                               | plane[bit + 2] << 1
                               | plane[bit + 3] << 0;
                out << std::hex << hexdigit;
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
            if (std::next(it) != end(step.probabilities)) {
                out << " ";
            }
        }
        out << std::endl;
        // And the game result for the side to move
        out << (step.to_move == BLACK ? -game_score : game_score) << std::endl;
    }
    outchunk.append(out.str());
}

void Training::dump_stats(const std::string& filename) {
    auto chunker = OutputChunker{filename, true};
    dump_stats(chunker);
}

void Training::dump_stats(OutputChunker& outchunk) {
    std::stringstream out;
    out << "1" << std::endl; // File format version 1
    for (const auto& step : m_data) {
        out << step.net_winrate
            << " " << step.root_uct_winrate
            << " " << step.child_uct_winrate
            << " " << step.bestmove_visits << std::endl;
    }
    outchunk.append(out.str());
}
