add_library (chess SHARED
	bitboard.cc
	board.cc
	position.cc
	bitboard.h
	board.h
	position.h
)                                                                                                                                 

if (GTEST_FOUND)
	file (GLOB TEST_SOURCES test/*)
	add_executable (chess_test ${TEST_SOURCES})

	target_link_libraries (chess_test
		chess
		${GTEST_BOTH_LIBRARIES}
		${CMAKE_THREAD_LIBS_INIT}
	)

	add_test (chess_test ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/chess_test)
endif (GTEST_FOUND)