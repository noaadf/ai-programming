CXX = clang++
CXXFLAGS = -std=c++17 -Wall

SRC = main.cpp tensor.cpp activation.cpp
OBJ = $(SRC:.cpp=.o)

main: $(OBJ)
	$(CXX) $(CXXFLAGS) -o main $(OBJ)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) main
