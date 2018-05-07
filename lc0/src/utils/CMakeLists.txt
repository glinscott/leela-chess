add_library (utils SHARED
    bititer.h
    cache-old.h
    cache.h
    commandline.h
    cppattributes.h
    exception.h
    hashcat.h
    mutex.h
    optional.h
    optionsdict.h
    random.h
    smallarray.h
    transpose.h
    commandline.cc
    optionsdict.cc
    random.cc
    transpose.cc
)                                                                                                                                 

if (GTEST_FOUND)
    file (GLOB TEST_SOURCES test/*)
    add_executable (utils_test ${TEST_SOURCES})

    target_link_libraries (utils_test
        utils
        ${GTEST_BOTH_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
    )

    add_test (utils_test ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/utils_test)
endif (GTEST_FOUND)