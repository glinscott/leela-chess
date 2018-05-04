# This file is part of Leela Zero.
# Copyright (C) 2017 Marco Calignano
# Copyright (C) 2017 Gian-Carlo Pascutto
# Leela Zero is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# Leela Zero is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

project (leela-chess CXX C)
cmake_minimum_required (VERSION 3.1)

set (CMAKE_CXX_STANDARD 14)
list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

add_subdirectory(gtest EXCLUDE_FROM_ALL) # We don't want to install gtest, exclude it from `all`

# Options
option (ENABLE_POPCOUNT "Enable builtin popcount support" OFF) # Ideally we'd autodetect

# Required Packages
set (Boost_MIN_VERSION "1.54.0")
set (Boost_USE_MULTITHREADED ON)
find_package (Boost 1.54.0 REQUIRED program_options filesystem)
find_package (Threads REQUIRED)
find_package (ZLIB REQUIRED)
if (NOT FEATURE_USE_CPU_ONLY)
    find_package (OpenCL REQUIRED)
endif()

# We need OpenBLAS for now, because we make some specific
# calls. Ideally we'd use OpenBLAS is possible and fall back to
# not doing those calls if it's not present.
if (NOT APPLE)
  set (BLA_VENDOR OpenBLAS)
endif ()

find_package (BLAS REQUIRED)
find_path (BLAS_INCLUDE_DIRS openblas_config.h
  /usr/include
  /usr/local/include
  /usr/include/openblas
  /opt/OpenBLAS/include
  /usr/include/x86_64-linux-gnu
  $ENV{BLAS_HOME}/include)

# See if we can set optimization flags as expected.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set (GccSpecificFlags 1)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  set (GccSpecificFlags 1)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set (GccSpecificFlags 1)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set (GccSpecificFlags 0)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set (GccSpecificFlags 0)
endif()

if (GccSpecificFlags)
  set (CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -pipe")
  set (CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
  set (CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto -ffast-math -march=native -mtune=native")
  set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
  set (CMAKE_EXE_LINKER_FLAGS "-flto")
  if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "RelWithDebInfo")
  endif (NOT CMAKE_BUILD_TYPE)
endif(GccSpecificFlags)

if (ENABLE_POPCOUNT)
  add_definitions (-DUSE_POPCNT)
endif()

if (FEATURE_USE_CPU_ONLY)
  add_definitions (-DFEATURE_USE_CPU_ONLY)
endif()

set (IncludePath "${CMAKE_CURRENT_SOURCE_DIR}/src")
set (SrcPath "${CMAKE_CURRENT_SOURCE_DIR}/src")

include_directories (${IncludePath})
include_directories (SYSTEM ${Boost_INCLUDE_DIRS})
if (NOT FEATURE_USE_CPU_ONLY)
    include_directories (SYSTEM ${OpenCL_INCLUDE_DIRS})
endif()
include_directories (SYSTEM ${ZLIB_INCLUDE_DIRS})

if ((UNIX AND NOT APPLE) OR WIN32)
  include_directories (${BLAS_INCLUDE_DIRS})
endif ()
if (APPLE)
  include_directories ("/System/Library/Frameworks/Accelerate.framework/Versions/Current/Headers")
endif ()

file (GLOB lczero_SRC "${SrcPath}/*.cpp" "${SrcPath}/*.h" "${SrcPath}/syzygy/*.cpp" "${SrcPath}/syzygy/*.h")

set (lczero_MAIN "${SrcPath}/main.cpp")
file (GLOB lczero_SRC "${SrcPath}/*.cpp" "${SrcPath}/syzygy/*.cpp")
list (REMOVE_ITEM lczero_SRC ${lczero_MAIN})

# Reuse for lczero and gtest
add_library (objs OBJECT ${lczero_SRC} src/Misc.h)
add_executable (lczero $<TARGET_OBJECTS:objs> ${lczero_MAIN})

target_link_libraries (lczero ${Boost_LIBRARIES})
target_link_libraries (lczero ${BLAS_LIBRARIES})
if (NOT FEATURE_USE_CPU_ONLY)
    target_link_libraries (lczero ${OpenCL_LIBRARIES})
endif()
target_link_libraries (lczero ${ZLIB_LIBRARIES})
target_link_libraries (lczero ${CMAKE_THREAD_LIBS_INIT})
install (TARGETS lczero DESTINATION bin)

# Google Test below
file (GLOB tests_SRC "${SrcPath}/tests/*.cpp")
add_executable (tests ${tests_SRC} $<TARGET_OBJECTS:objs>)

target_link_libraries (tests ${Boost_LIBRARIES})
target_link_libraries (tests ${BLAS_LIBRARIES})
if (NOT FEATURE_USE_CPU_ONLY)
    target_link_libraries (tests ${OpenCL_LIBRARIES})
endif()
target_link_libraries (tests ${ZLIB_LIBRARIES})
target_link_libraries (tests gtest_main ${CMAKE_THREAD_LIBS_INIT})
