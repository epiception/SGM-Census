INC=`pkg-config --cflags --libs opencv`
INC_DIR=progressbar
DEPS=./$(INC_DIR)/ProgressBar.h ./$(INC_DIR)/ProgressBar.cc
CXXFLAGS= -o3 -std=c++11 -I$(INC_DIR)

SOURCES=sgm.cpp
EXECUTABLE=sgm

all:
	g++ $(CXXFLAGS) $(DEPS) $(SOURCES) -o $(EXECUTABLE) $(INC)
