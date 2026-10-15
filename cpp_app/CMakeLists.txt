cmake_minimum_required(VERSION 3.16)
project(ActionRecognitionApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)

# ---------------------------------------------------------------------------
# Qt 6
# ---------------------------------------------------------------------------
find_package(Qt6 REQUIRED COMPONENTS Widgets Core)

# ---------------------------------------------------------------------------
# OpenCV (manual setup to bypass broken auto-detection on VS2026)
# ---------------------------------------------------------------------------
set(OpenCV_MANUAL_DIR "C:/OpenCV/opencv/build" CACHE PATH "Root of the prebuilt OpenCV Windows package")
set(OpenCV_INCLUDE_DIRS "${OpenCV_MANUAL_DIR}/include")

file(GLOB OpenCV_ALL_LIBS "${OpenCV_MANUAL_DIR}/x64/vc16/lib/*.lib")
set(OpenCV_RELEASE_LIBS "")
set(OpenCV_DEBUG_LIBS "")
foreach(lib_path ${OpenCV_ALL_LIBS})
    get_filename_component(lib_name "${lib_path}" NAME_WE)
    if(lib_name MATCHES "d$")
        list(APPEND OpenCV_DEBUG_LIBS "${lib_path}")
    else()
        list(APPEND OpenCV_RELEASE_LIBS "${lib_path}")
    endif()
endforeach()

if(NOT OpenCV_RELEASE_LIBS)
    message(FATAL_ERROR "No OpenCV .lib files found in ${OpenCV_MANUAL_DIR}/x64/vc16/lib - check the path.")
endif()

# ---------------------------------------------------------------------------
# ONNX Runtime (manual setup - same pattern as OpenCV above)
# Download from: https://github.com/microsoft/onnxruntime/releases
# Pick:  onnxruntime-win-x64-<version>.zip  (NOT GPU, NOT training)
# Extract to C:/onnxruntime  so you get C:/onnxruntime/include, lib, bin
# ---------------------------------------------------------------------------
set(ORT_DIR "C:/onnxruntime" CACHE PATH "ONNX Runtime root directory")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_IMPLIB             "${ORT_DIR}/lib/onnxruntime.lib"
    IMPORTED_LOCATION           "${ORT_DIR}/lib/onnxruntime.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include"
)

# ---------------------------------------------------------------------------
# Application target
# ---------------------------------------------------------------------------
add_executable(ActionRecognitionApp
    main.cpp
    webcamwidget.cpp   webcamwidget.h
    pose_extractor.cpp pose_extractor.h
    action_inference.cpp action_inference.h
    object_renderer.cpp object_renderer.h
    normalizer.h
)

target_include_directories(ActionRecognitionApp PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ActionRecognitionApp PRIVATE
    Qt6::Widgets
    Qt6::Core
    onnxruntime
    $<$<CONFIG:Debug>:${OpenCV_DEBUG_LIBS}>
    $<$<NOT:$<CONFIG:Debug>>:${OpenCV_RELEASE_LIBS}>
)

# ---------------------------------------------------------------------------
# Copy onnxruntime.dll next to the .exe automatically after every build
# so you don't have to add C:/onnxruntime/bin to PATH manually
# ---------------------------------------------------------------------------
add_custom_command(TARGET ActionRecognitionApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_DIR}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:ActionRecognitionApp>"
    COMMENT "Copying onnxruntime.dll to output folder"
)