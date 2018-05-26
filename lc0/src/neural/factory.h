/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <functional>
#include <string>
#include "neural/network.h"
#include "utils/optionsdict.h"

namespace lczero {

class NetworkFactory {
 public:
  using FactoryFunc = std::function<std::unique_ptr<Network>(
      const Weights&, const OptionsDict&)>;

  static NetworkFactory* Get();

  // Registers network so it can be created by name.
  // @name -- name
  // @options -- options to pass to the network
  // @priority -- how high should be the network in the list. The network with
  //              the highest priority is the default.
  class Register {
   public:
    Register(const std::string& name, FactoryFunc factory, int priority = 0);
  };

  // Returns list of backend names, sorted by priority (higher priority first).
  std::vector<std::string> GetBackendsList() const;

  // Creates a backend given name and config.
  std::unique_ptr<Network> Create(const std::string& network, const Weights&,
                                  const OptionsDict& options);

 private:
  void RegisterNetwork(const std::string& name, FactoryFunc factory,
                       int priority);

  NetworkFactory() {}

  struct Factory {
    Factory(const std::string& name, FactoryFunc factory, int priority)
        : name(name), factory(factory), priority(priority) {}

    bool operator<(const Factory& other) const {
      if (priority != other.priority) return priority > other.priority;
      return name < other.name;
    }

    std::string name;
    FactoryFunc factory;
    int priority;
  };

  std::vector<Factory> factories_;
  friend class Register;
};

#define REGISTER_NETWORK_WITH_COUNTER2(name, cls, priority, counter) \
  namespace {                                                        \
  static NetworkFactory::Register regH38fhs##counter(                \
      name,                                                          \
      [](const Weights& w, const OptionsDict& o) {                   \
        return std::make_unique<cls>(w, o);                          \
      },                                                             \
      priority);                                                     \
  }
#define REGISTER_NETWORK_WITH_COUNTER(name, cls, priority, counter) \
  REGISTER_NETWORK_WITH_COUNTER2(name, cls, priority, counter)

// Registers a Network.
// Constructor of a network class must have parameters:
// (const Weights& w, const OptionsDict& o)
// @name -- name under which the backend will be known in configs.
// @cls -- class name of a backend.
// @priority -- numeric priority of a backend. Higher is higher, highest number
// is the default backend.
#define REGISTER_NETWORK(name, cls, priority) \
  REGISTER_NETWORK_WITH_COUNTER(name, cls, priority, __LINE__)
}  // namespace lczero
