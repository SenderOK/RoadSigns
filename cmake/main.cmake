SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )

SET(MAIN ${Source_Path}/main.cpp
         ${Source_Path}/classifier.h
         ${Source_Path}/matrix.hpp
         ${Source_Path}/matrix.h
         ${Source_Path}/io.hpp
         ${Source_Path}/io.h
         ${Source_Path}/filters.h
         ${Source_Path}/filters.cpp)
