objects = main.o LapPym.o MultiFusion.o
CFLAGS = `pkg-config --cflags --libs opencv`
LIBS = `pkg-config --libs opencv`
CPPFLAGS = -std=c++11

out : $(objects)
	g++ $(CPPFLAGS) -o out $(objects) $(CFLAGS) $(LIBS)

main.o : Fusion/MultiFusion.h
	g++ $(CPPFLAGS) -c main.cpp $(CFLAGS) $(LIBS)
MultiFusion.o : Fusion/MultiFusion.h Fusion/LapPym.hpp
	g++ $(CPPFLAGS) -c Fusion/MultiFusion.cpp $(CFLAGS) $(LIBS)
LapPym.o : Fusion/LapPym.hpp
	g++ $(CPPFLAGS) -c Fusion/LapPym.cpp $(CFLAGS) $(LIBS)

.PHONY : clean

clean : 
	-rm out $(objects)
