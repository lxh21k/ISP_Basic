#1. cmake version
cmake_minimum_required(VERSION 3.2)
#2. project name
SET(PROJECT_NAME yuv2rgb)
PROJECT(${PROJECT_NAME})

#3. head file path
FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(include)

#4. source directory

AUX_SOURCE_DIRECTORY(src DIR_SRCS)

#5.set environment variable
SET(SRC ${DIR_SRCS})

#6. add executable

ADD_EXECUTABLE(${PROJECT_NAME} ${SRC})

#7. add link library

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})

