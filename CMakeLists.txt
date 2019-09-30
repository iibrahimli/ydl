cmake_minimum_required(VERSION 3.10)

project(ydl)



# add Darknet

set(DARKNET_PATH ${PROJECT_SOURCE_DIR}/darknet)

set(DARKNET_LIB ${DARKNET_PATH}/libdarknet.so)

include_directories(${DARKNET_PATH}/include)

add_custom_target(
    libdarknet
    COMMAND make
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/darknet
)



# test executable linked against Darknet

add_executable(ydl_test ${PROJECT_SOURCE_DIR}/src/ydl_test.cpp)

target_link_libraries(ydl_test ${DARKNET_LIB})

add_dependencies(ydl_test libdarknet)