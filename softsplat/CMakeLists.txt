# message(STATUS "MNN Depends:" ${MNN_DEPS})
set(SVFI_SOFTSPLAT_DEMO "")

find_package(ZLIB REQUIRED)

# 添加cnpy头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/cnpy)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/zlib)
include_directories(${MNN_INCLUDE_DIRS})

add_executable(softsplat_demo.out 
    ${CMAKE_CURRENT_LIST_DIR}/softsplat_demo.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/cnpy/cnpy.cpp
    )
list(APPEND SVFI_SOFTSPLAT_DEMO softsplat_demo.out)

foreach(TARGET ${SVFI_SOFTSPLAT_DEMO})
    target_link_libraries(${TARGET} ${MNN_DEPS})
    target_link_libraries(${TARGET} ${ZLIB_LIBRARIES})
    if (MSVC)
        target_compile_definitions(${TARGET} PRIVATE "_CRT_SECURE_NO_WARNINGS")
        if (NOT MNN_BUILD_SHARED_LIBS)
            foreach (DEPEND ${MNN_DEPS})
                target_link_options(${TARGET} PRIVATE /WHOLEARCHIVE:$<TARGET_FILE:${DEPEND}>)
            endforeach ()
        endif()
    endif()
endforeach()
