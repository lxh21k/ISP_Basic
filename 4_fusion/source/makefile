objects = main.o LapPym.o MultiFusion.o
CFLAGS = `pkg-config --cflags --libs opencv4`
LIBS = `pkg-config --libs opencv4`
CPPFLAGS = -std=c++11

mout : $(objects)
	g++ $(CPPFLAGS) -o mout $(objects) $(CFLAGS) $(LIBS)

main.o : Fusion/MultiFusion.h
	g++ $(CPPFLAGS) -c main.cpp $(CFLAGS) $(LIBS)
MultiFusion.o : Fusion/MultiFusion.h Fusion/LapPym.hpp
	g++ $(CPPFLAGS) -c Fusion/MultiFusion.cpp $(CFLAGS) $(LIBS)
LapPym.o : Fusion/LapPym.hpp
	g++ $(CPPFLAGS) -c Fusion/LapPym.cpp $(CFLAGS) $(LIBS)

.PHONY : clean

clean : 
	-rm mout $(objects)
