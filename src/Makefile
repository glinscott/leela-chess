THE_OS := $(shell uname -s)

default:
	@echo "Detected OS: ${THE_OS}"
	$(MAKE) \
		CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -pipe -O3 -g -ffast-math -flto -march=native -std=c++14 -DNDEBUG'  \
		LDFLAGS='$(LDFLAGS) -flto -g' \
		lczero

#CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -pipe -O0 -g -std=c++14 -fsanitize=thread'
#LDFLAGS='$(LDFLAGS) -g -fsanitize=thread -fuse-ld=gold'

debug:
	@echo "Detected OS: ${THE_OS}"
	$(MAKE) \
		CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -pipe -O0 -g -std=c++14' \
		LDFLAGS='$(LDFLAGS) -g' \
		lczero

clang:
	@echo "Detected OS: ${THE_OS}"
	$(MAKE) CC=clang-5.0 CXX=clang++-5.0 \
		CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -Wno-missing-braces -O3 -ffast-math -flto -march=native -std=c++14 -DNDEBUG' \
		LDFLAGS='$(LDFLAGS) -flto -fuse-linker-plugin' \
		lczero

DYNAMIC_LIBS = -lboost_program_options -lboost_system -lboost_filesystem -lpthread -lz
LIBS =

ifeq ($(THE_OS),Linux)
# for Linux with OpenBLAS
	CXXFLAGS += -I/usr/include/openblas
	DYNAMIC_LIBS += -lopenblas
	DYNAMIC_LIBS += -lOpenCL
endif
ifeq ($(THE_OS),Darwin)
# for macOS (comment out the Linux part)
	LIBS += -framework Accelerate
	LIBS += -framework OpenCL
	CXXFLAGS += -I/System/Library/Frameworks/Accelerate.framework/Versions/Current/Headers -Wno-deprecated-declarations
endif

# for MKL instead of OpenBLAS
#DYNAMIC_LIBS += -lmkl_rt
#CXXFLAGS += -I/opt/intel/mkl/include
#LDFLAGS  += -L/opt/intel/mkl/lib/intel64/

CXXFLAGS += -I. -Wno-ignored-attributes
CPPFLAGS += -MD -MP

sources = Network.cpp Training.cpp UCTSearch.cpp Utils.cpp Random.cpp Parameters.cpp \
		UCTNode.cpp OpenCL.cpp Timing.cpp main.cpp Tuner.cpp \
		Bitboard.cpp Movegen.cpp Position.cpp UCI.cpp pgn.cpp SMP.cpp OpenCLScheduler.cpp \
		NNCache.cpp TimeMan.cpp UCIOption.cpp syzygy/tbprobe.cpp

objects = $(sources:.cpp=.o)
deps = $(sources:%.cpp=%.d)

-include $(deps)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

lczero: $(objects)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS) $(DYNAMIC_LIBS)

clean:
	-$(RM) lczero $(objects) $(deps)

.PHONY: clean default debug clang
