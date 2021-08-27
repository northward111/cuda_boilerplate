cmake_minimum_required(VERSION 3.19)
set(PROJECT_NAME "cuda_boilerplate")
project("${PROJECT_NAME}" LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

set(PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
# 设置库文件生成目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_DIR}/bin)

# 设置lib目录
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party_win64")
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    file(GLOB DLL_LIST "${THIRD_PARTY_DIR}/*/lib/debug/*.dll")
else()
    file(GLOB DLL_LIST "${THIRD_PARTY_DIR}/*/lib/release/*.dll")
endif()
file(COPY ${DLL_LIST} DESTINATION "${PROJECT_DIR}/bin")
file(GLOB DLL_LIST "${THIRD_PARTY_DIR}/*/lib/*.dll")
file(COPY ${DLL_LIST} DESTINATION "${PROJECT_DIR}/bin")

# 设置头文件
INCLUDE_DIRECTORIES("${PROJECT_DIR}")
FILE(GLOB third_party_sub_dirs "${THIRD_PARTY_DIR}/*")
foreach(third_party_sub_dir ${third_party_sub_dirs})
    INCLUDE_DIRECTORIES("${third_party_sub_dir}/include")
endforeach()
INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# 设置lib目录
FILE(GLOB third_party_sub_dirs "${THIRD_PARTY_DIR}/*")
foreach(third_party_sub_dir ${third_party_sub_dirs})
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        link_directories("${third_party_sub_dir}/lib/debug")
    else()
        link_directories("${third_party_sub_dir}/lib/release")
    endif()
    link_directories("${third_party_sub_dir}/lib")
endforeach()
link_directories("${PROJECT_DIR}/bin")

add_executable("${PROJECT_NAME}" main.cpp kernel.cu)

