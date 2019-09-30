cmake_minimum_required(VERSION 3.10)

project(ydl VERSION 0.1 LANGUAGES C CXX)



# add Darknet

set(DARKNET_PATH ${PROJECT_SOURCE_DIR}/darknet)

set(DARKNET_LIB ${DARKNET_PATH}/libdarknet.so)

include_directories(${DARKNET_PATH}/include)

add_custom_target(
    libdarknet
    COMMAND make
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/darknet
)



# library

set(YDL_SOURCES
        src/load_model.cpp
)

add_library(${PROJECT_NAME} STATIC ${YDL_SOURCES})

# add_library(${PROJECT_NAME} SHARED ${YDL_SOURCES})



# test executable linked against Darknet

add_executable(ydl_test ${PROJECT_SOURCE_DIR}/src/ydl_test.cpp)

target_link_libraries(ydl_test ${DARKNET_LIB})

add_dependencies(ydl_test libdarknet)