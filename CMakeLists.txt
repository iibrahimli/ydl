cmake_minimum_required(VERSION 3.10)

project(ydl VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# add Darknet

set(DARKNET_PATH ${PROJECT_SOURCE_DIR}/darknet)

set(DARKNET_LIB ${DARKNET_PATH}/libdarknet.so)

include_directories(${DARKNET_PATH}/include)

add_custom_target(
    libdarknet
    COMMAND make
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/darknet
)



# add OpenCV
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


# export ydl library

set(YDL_SOURCES
        src/detector.cpp
)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/ydl)

add_library(${PROJECT_NAME} STATIC ${YDL_SOURCES})

# add_library(${PROJECT_NAME} SHARED ${YDL_SOURCES})



# test executable linked against Darknet

add_executable(ydl_test ${PROJECT_SOURCE_DIR}/src/ydl_test.cpp)

target_link_libraries(ydl_test ${PROJECT_NAME} ${DARKNET_LIB} ${OpenCV_LIBS})

add_dependencies(ydl_test libdarknet ${PROJECT_NAME})