##########################################################################################
# CMake build script for task 2 of Machine Graphics course.
# Copyright Moscow State University, Graphics and Media Lab.
##########################################################################################

PROJECT(Task2 CXX C)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

IF(NOT CMAKE_BUILD_TYPE)
    MESSAGE(STATUS "No build type selected, default is Debug")
    SET(CMAKE_BUILD_TYPE "Debug")
ENDIF()

IF(NOT DEFINED CMAKE_CURRENT_LIST_DIR)
    SET(CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
ENDIF(NOT DEFINED CMAKE_CURRENT_LIST_DIR)

MESSAGE(STATUS "CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}")
MESSAGE(STATUS "PROJECT_BINARY_DIR    : ${PROJECT_BINARY_DIR}")

SET(Source_Path ${CMAKE_CURRENT_LIST_DIR})

MESSAGE(STATUS "Source_Path           : ${Source_Path}")

SET(CMAKE_DEBUG_POSTFIX "_d")
MESSAGE(STATUS "CMAKE_DEBUG_POSTFIX   : ${CMAKE_DEBUG_POSTFIX}")

IF(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wno-unused-result")
ENDIF(UNIX)

##########################################################################################
# Adding executable of project

INCLUDE(${Source_Path}/cmake/main.cmake)
INCLUDE(${Source_Path}/cmake/argvparser.cmake)
INCLUDE(${Source_Path}/cmake/EasyBMP.cmake)
INCLUDE(${Source_Path}/cmake/liblinear.cmake)
ADD_EXECUTABLE(task2 ${MAIN} ${ARGVPARSER} ${EASY_BMP} ${LIBLINEAR})

##########################################################################################
# Adding executable of test program

INCLUDE(${Source_Path}/cmake/test.cmake)
ADD_EXECUTABLE(task2_test ${TEST} ${ARGVPARSER})
