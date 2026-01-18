cmake_minimum_required(VERSION 3.22.1)
project("sd_native")

# 自动定位 .so 库的绝对目录
set(MNN_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})

# 导入 MNN 核心库
add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION ${MNN_LIB_DIR}/libMNN.so)

add_library(MNN_Express SHARED IMPORTED)
set_target_properties(MNN_Express PROPERTIES IMPORTED_LOCATION ${MNN_LIB_DIR}/libMNN_Express.so)

add_library(MNN_LLM SHARED IMPORTED)
set_target_properties(MNN_LLM PROPERTIES IMPORTED_LOCATION ${MNN_LIB_DIR}/libllm.so)

# 你的业务 JNI 库
add_library(sd_engine SHARED native-lib.cpp)

# 确保你已经把 MNN 源码的 include 文件夹放入 cpp 目录
target_include_directories(sd_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 链接
target_link_libraries(sd_engine
        MNN_Express
        MNN
        log
        jnigraphics
)