cmake_minimum_required(VERSION 3.22.1)
project("sd_native")

# 1. 自动定位 .so 库
set(MNN_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})

# 2. 导入 MNN 核心及 Express (OpenGL 模式下建议也加载 CL 以增强驱动可见性)
add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION ${MNN_LIB_DIR}/libMNN.so)

add_library(MNN_Express SHARED IMPORTED)
set_target_properties(MNN_Express PROPERTIES IMPORTED_LOCATION ${MNN_LIB_DIR}/libMNN_Express.so)

# 3. 定义你的业务库
add_library(sd_engine SHARED native-lib.cpp)

# 【核心修改 1】适配 Android 15+ 骁龙 8 Elite 必做：16KB Page Size
# 官方源码明确指出这是解决新设备崩溃的关键
target_link_options(sd_engine PRIVATE "-Wl,-z,max-page-size=16384")

# 4. 头文件路径
target_include_directories(sd_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 5. 链接所有库
# 移除 vulkan，补全官方链接的 android, log 和 mediandk
# 显式链接 GLESv3 和 EGL 以确保荣耀系统能唤醒 OpenGL 加速
target_link_libraries(sd_engine
        android      # 官方系统库
        log          # 官方系统库
        mediandk     # 官方系统库
        GLESv3       # OpenGL ES 3.0 系统驱动
        EGL          # OpenGL 环境管理库
        MNN
        MNN_Express
        jnigraphics
)