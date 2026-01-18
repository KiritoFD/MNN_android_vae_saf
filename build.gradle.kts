// 路径: /build.gradle.kts (根目录)
plugins {
    // 这里只定义版本，不实际应用，防止 Extension 冲突
    id("com.android.application") version "8.7.0" apply false
    id("com.android.library") version "8.7.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.22" apply false
}