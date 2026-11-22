CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -fPIC
build/libfengine.so: src/fengine.cpp src/capi.cpp src/fengine.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -shared -o $@ src/fengine.cpp src/capi.cpp
clean:
	rm -rf build
