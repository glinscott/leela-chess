/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2018 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad
  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <cassert>
#include <ostream>

#include "Utils.h"
#include "UCI.h"
#include "Parameters.h"

using std::string;

using namespace Utils;

UCI::OptionsMap Options; // Global object

namespace UCI {

/// 'On change' actions, triggered by an option's value change
    void on_threads(const Option& o) {
        int num_threads = o;

        if (num_threads > cfg_num_threads) {
            for (auto i = thread_pool.size(); i < static_cast<std::size_t>(num_threads); ++i) {
                thread_pool.add_thread([](){});
            }

            cfg_num_threads = num_threads;
        } else {
            cfg_num_threads = num_threads; //we just decrease the number of threads used in the search loop
        }

        myprintf("Using %d thread(s).\n", num_threads);
    }

    void on_quiet(const Option& o) {
        bool value = o;

        if (value) {
            myprintf("Enabled quiet mode\n", value);
        } else {
            myprintf("Disabled quiet mode\n", value);
        }

        cfg_quiet = value;
    }

    bool set_float_cfg(float& cfg_param, const std::string& value) {
        try {
            cfg_param = std::strtof(value.c_str(), nullptr);
        } catch (const std::logic_error& exc) {
            myprintf("Could not convert to float: %s\n", value.c_str());

            return false;
        }

        return true;
    }

    void on_softmaxtemp(const Option& o) {
        std::string value = o;

        if (set_float_cfg(cfg_softmax_temp, value)) {
            myprintf("Set cfg_softmax_temp to %.6f\n", cfg_softmax_temp);
        }
    }

    void on_fpureduction(const Option& o) {
        std::string value = o;

        if (set_float_cfg(cfg_fpu_reduction, value)) {
            myprintf("Set cfg_fpu_reduction to %.6f\n", cfg_fpu_reduction);
        }
    }

    void on_fpudynamiceval(const Option& o) {
        bool value = o;

        cfg_fpu_dynamic_eval = value;

        if (value) {
            myprintf("cfg_fpu_dynamic_eval enabled\n");
        } else {
            myprintf("cfg_fpu_dynamic_eval disabled\n");
        }
    }

    void on_puct(const Option& o) {
        std::string value = o;

        if (set_float_cfg(cfg_puct, value)) {
            myprintf("Set puct to %.6f\n", cfg_puct);
        }
    }

    void on_slowmover(const Option& o) {
        int value = o;

        cfg_slowmover = value;
        myprintf("Set cfg_slowmover to %d.\n", cfg_slowmover);
    }

    void on_nodes_as_visits(const Option& o) {
        cfg_go_nodes_as_visits = o;

        if (cfg_go_nodes_as_visits) {
            myprintf("Set go nodes to visits.\n");
        } else {
            myprintf("Set go nodes to playouts.\n");
        }
    }

/// Our case insensitive less() function as required by UCI protocol
    bool CaseInsensitiveLess::operator() (const string& s1, const string& s2) const {

        return std::lexicographical_compare(s1.begin(), s1.end(), s2.begin(), s2.end(),
                                            [](char c1, char c2) { return tolower(c1) < tolower(c2); });
    }


/// init() initializes the UCI options to their hard-coded default values

    void init(OptionsMap& o) {
        o["Threads"]                << Option(cfg_num_threads, 1, cfg_max_threads, on_threads);
        o["Quiet"]                  << Option(cfg_quiet, on_quiet);
        o["Softmax Temp"]           << SilentOption(std::to_string(cfg_softmax_temp).c_str(), on_softmaxtemp);
        o["FPU Reduction"]          << Option(std::to_string(cfg_fpu_reduction).c_str(), on_fpureduction);
        o["FPU Dynamic Eval"]       << SilentOption(cfg_fpu_dynamic_eval, on_fpudynamiceval);
        o["Puct"]                   << Option(std::to_string(cfg_puct).c_str(), on_puct);
        o["SlowMover"]              << Option(cfg_slowmover, 1, std::numeric_limits<int>::max(), on_slowmover);
        o["Go Nodes Visits"]        << Option(cfg_go_nodes_as_visits, on_nodes_as_visits);
    }

/// operator<<() is used to print all the options default values in chronological
/// insertion order (the idx field) and in the format defined by the UCI protocol.

    std::ostream& operator<<(std::ostream& os, const OptionsMap& om) {

        for (size_t idx = 0; idx < om.size(); ++idx)
            for (const auto& it : om)
                if (it.second.idx == idx && it.second.advertise)
                {
                    const Option& o = it.second;

                    os << "\noption name " << it.first << " type " << o.type;

                    if (o.type != "button")
                        os << " default " << o.defaultValue;

                    if (o.type == "spin")
                        os << " min " << o.min << " max " << o.max;

                    break;
                }

        return os;
    }


/// Option class constructors and conversion operators

    Option::Option(const char* v, OnChange f) : type("string"), min(0), max(0), on_change(f)
    { defaultValue = currentValue = v; }

    Option::Option(bool v, OnChange f) : type("check"), min(0), max(0), on_change(f)
    { defaultValue = currentValue = (v ? "true" : "false"); }

    Option::Option(OnChange f) : type("button"), min(0), max(0), on_change(f)
    {}

    Option::Option(int v, int minv, int maxv, OnChange f) : type("spin"), min(minv), max(maxv), on_change(f)
    { defaultValue = currentValue = std::to_string(v); }

    Option::operator int() const {
        assert(type == "check" || type == "spin");
        return (type == "spin" ? stoi(currentValue) : currentValue == "true");
    }

    Option::operator std::string() const {
        assert(type == "string");
        return currentValue;
    }


/// operator<<() inits options and assigns idx in the correct printing order

    void Option::operator<<(const Option& o) {

        static size_t insert_order = 0;

        *this = o;
        idx = insert_order++;
    }


/// operator=() updates currentValue and triggers on_change() action. It's up to
/// the GUI to check for option's limits, but we could receive the new value from
/// the user by console window, so let's check the bounds anyway.

    Option& Option::operator=(const string& v) {

        assert(!type.empty());

        if (   (type != "button" && v.empty())
               || (type == "check" && v != "true" && v != "false")
               || (type == "spin" && (stoi(v) < min || stoi(v) > max)))
            return *this;

        if (type != "button")
            currentValue = v;

        if (on_change)
            on_change(*this);

        return *this;
    }

} // namespace UCI
