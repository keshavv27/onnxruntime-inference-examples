cmake_minimum_required(VERSION 3.13)
project(stablediffusion VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


option(onnxruntime_USE_NV "Build with NvTensortRtx support" OFF)
option(ONNXRUNTIME_ROOT_PATH "onnxruntime root dir")

if(NOT ONNXRUNTIME_ROOT_PATH)
  set(ONNXRUNTIME_ROOT_PATH "C:/Program Files/onnxruntime")
endif()
include_directories("${ONNXRUNTIME_ROOT_PATH}/include" "${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/")
link_directories("${ONNXRUNTIME_ROOT_PATH}/lib")


# Check if we're on Windows
if(WIN32)
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT_PATH}/include")
    set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT_PATH}/lib")

    # Find the library
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS ${ONNXRUNTIME_LIB_DIR}
        REQUIRED
    )
else()
    # Linux/Unix
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT_PATH}/include")
    set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT_PATH}/lib")

    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS ${ONNXRUNTIME_LIB_DIR}
        REQUIRED
    )
endif()

if(onnxruntime_USE_NV)
  add_definitions(-DUSE_NV)
endif()

# Create executable
add_executable(sd_pipeline
    main.cpp
    stable_diffusion_pipeline.cpp
)

# Include directories
target_include_directories(sd_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(sd_pipeline PRIVATE
    ${ONNXRUNTIME_LIB}
)

# Platform-specific settings
if(WIN32)
    # Windows specific settings
    target_compile_definitions(sd_pipeline PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
    )

    # Copy ONNX Runtime DLL to output directory
    get_filename_component(ONNXRUNTIME_DLL_NAME ${ONNXRUNTIME_LIB} NAME_WE)
    set(ONNXRUNTIME_DLL "${ONNXRUNTIME_LIB_DIR}/${ONNXRUNTIME_DLL_NAME}.dll")

    if(EXISTS ${ONNXRUNTIME_DLL})
        add_custom_command(TARGET sd_pipeline POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNXRUNTIME_DLL}
            $<TARGET_FILE_DIR:sd_pipeline>
            COMMENT "Copying ONNX Runtime DLL"
        )
    endif()

    # Additional Windows libraries
    target_link_libraries(sd_pipeline PRIVATE
        kernel32
        user32
        gdi32
        winspool
        comdlg32
        advapi32
        shell32
        ole32
        oleaut32
        uuid
        odbc32
        odbccp32
    )
else()
    # Linux/Unix specific settings
    target_compile_options(sd_pipeline PRIVATE
        -Wall
        -Wextra
        -O3
        -march=native
    )

    # Link pthread for threading support
    find_package(Threads REQUIRED)
    target_link_libraries(sd_pipeline PRIVATE Threads::Threads)

    # Link dl for dynamic linking
    target_link_libraries(sd_pipeline PRIVATE dl)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(sd_pipeline PRIVATE
        -Wno-unused-parameter
        -Wno-unused-variable
    )
endif()

# Build type specific settings
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(WIN32)
        target_compile_options(sd_pipeline PRIVATE /O2 /DNDEBUG)
    else()
        target_compile_options(sd_pipeline PRIVATE -O3 -DNDEBUG)
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(WIN32)
        target_compile_options(sd_pipeline PRIVATE /Od /Zi)
    else()
        target_compile_options(sd_pipeline PRIVATE -O0 -g)
    endif()
endif()

# Print detailed build information
message(STATUS "ONNX Runtime Root: ${ONNXRUNTIME_ROOT_PATH}")
message(STATUS "ONNX Runtime Include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime Lib Dir: ${ONNXRUNTIME_LIB_DIR}")
message(STATUS "ONNX Runtime Library: ${ONNXRUNTIME_LIB}")

# Check if we can find version header
if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime/core/session/onnxruntime_cxx_api.h")
    message(STATUS "ONNX Runtime headers found")
else()
    message(WARNING "ONNX Runtime headers NOT found at expected location")
endif()

# Check library file
if(EXISTS "${ONNXRUNTIME_LIB}")
    message(STATUS "ONNX Runtime library file exists")
else()
    message(WARNING "ONNX Runtime library file NOT found: ${ONNXRUNTIME_LIB}")
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")

# Installation (optional)
install(TARGETS sd_pipeline
    RUNTIME DESTINATION bin
)

if(WIN32 AND EXISTS ${ONNXRUNTIME_DLL})
    install(FILES ${ONNXRUNTIME_DLL}
        DESTINATION bin
    )
endif()

