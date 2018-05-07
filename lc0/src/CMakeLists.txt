add_subdirectory (chess)
add_subdirectory (utils)
add_subdirectory (mcts)
add_subdirectory (selfplay)
add_subdirectory (neural)

add_executable (lc0
  engine.h
  optionsparser.h
  uciloop.h
  engine.cc
  main.cc
  optionsparser.cc
  uciloop.cc
)

target_link_libraries (lc0
  utils
  chess
  mcts
  neural
  selfplay
  ${CMAKE_THREAD_LIBS_INIT}
  ${Boost_LIBRARIES}
)
