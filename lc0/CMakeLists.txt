# This file is part of Leela Chess Zero.
# Copyright (C) 2018 The LCZero Authors
# 
# Leela Chess is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Leela Chess is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

project (lc0 CXX C)
cmake_minimum_required (VERSION 3.1)


#-------------------------------------------------------------------------------
# General Settings
#-------------------------------------------------------------------------------
set (CMAKE_CXX_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE "Release")
endif (NOT CMAKE_BUILD_TYPE)

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

list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

if (GccSpecificFlags)
    set (CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -pipe")
    set (CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set (CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -march=native -mtune=native")
    set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
    set (CMAKE_EXE_LINKER_FLAGS -Wl,--no-as-needed)
endif(GccSpecificFlags)


#-------------------------------------------------------------------------------
# Find packages
#-------------------------------------------------------------------------------
find_package (Threads REQUIRED)
find_package (Boost REQUIRED filesystem)
find_package (GTest)
find_package (CUDA)
find_package (CUDNN)

#-------------------------------------------------------------------------------
# Testing
#-------------------------------------------------------------------------------
if (GTEST_FOUND)
    enable_testing ()
endif (GTEST_FOUND)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

add_subdirectory (src)
