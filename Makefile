# A simple makefile for PIMC code

CC = mpiicpc
CFLAGS = -std=c++11
OPT = -O3 -ipo -fno-alias -xHost -unroll -no-prec-div
LIBS =
INC = 


EXE = som.x
DEP = $(shell find src/*.h)
SRC = $(shell find src/*.cpp)
OBJ = $(SRC:%.cpp=%.o)

%.o: %.cpp $(DEP)
	$(CC) $(DFLAGS) $(CFLAGS) $(OPT) -c $< -o $@ $(INC) $(LIBS)

$(EXE): $(OBJ)
	$(CC) $(DFLAGS) $(CFLAGS) $(OPT) -o $(EXE) $(OBJ) $(INC) $(LIBS) 
	rm -rf $(OBJ)

.PHONY: clean clean-all

clean:
	rm -rf $(OBJ)

clean-all:
	rm -rf $(OBJ) $(EXE)
