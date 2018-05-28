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


// Select the BLAS vendor based on defines

#ifdef USE_MKL
#include <mkl.h>
#else

#ifdef USE_OPENBLAS
#include <cblas.h>
#else

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define USE_ACCELERATE
#endif

#endif // USE_OPENBLAS

#endif // USE_MKL


