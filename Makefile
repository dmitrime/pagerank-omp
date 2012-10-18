CC=g++
CFLAGS=-march=native -std=c++0x -fopenmp -O3 -Wall
CFLAGS_D=-march=native -fopenmp -Wall -g

TARGET=PageRank

LIBS=-lrt

debug: 
	$(CC) -o $(TARGET) $(CFLAGS_D) pagerank.cpp $(LIBS) -DDEBUG
normal: 
	$(CC) -o $(TARGET) $(CFLAGS) pagerank.cpp $(LIBS)
clean:
	rm -f $(TARGET) 
