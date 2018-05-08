add_library (selfplay SHARED
  game.h
  loop.h
  tournament.h
  game.cc
  loop.cc
  tournament.cc
)                                                                                                                                 

target_link_libraries(selfplay
  chess
  utils
  mcts
  neural
)
