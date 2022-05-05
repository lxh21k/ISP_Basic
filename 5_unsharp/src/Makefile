objects = main.o adaptive_unsharp.o
CFLAGS = `pkg-config --cflags --libs opencv`
LIBS = `pkg-config --libs opencv`
CPPFLAGS = -std=c++11

out : $(objects)
	g++ $(CPPFLAGS) -o out $(objects) $(CFLAGS) $(LIBS)

main.o : adaptive_unsharp.h
	g++ $(CPPFLAGS) -c main.cpp $(CFLAGS) $(LIBS)
adaptive_unsharp.o : adaptive_unsharp.h
	g++ $(CPPFLAGS) -c adaptive_unsharp.cpp $(CFLAGS) $(LIBS)

.PHONY : clean
clean : 
	-rm out $(objects)