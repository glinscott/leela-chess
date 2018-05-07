add_library (mcts SHARED
  callbacks.h
  node.h
  search.h
  node.cc
  search.cc
)

target_link_libraries (mcts
  chess
)
