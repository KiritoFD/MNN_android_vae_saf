// 路径: /app/build.gradle.kts
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.sdapp"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.sdapp"
        // 维持 26 以解决 adaptive-icon 报错
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters.add("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                arguments("-DANDROID_STL=c++_shared")
                arguments("-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=16384")
                cppFlags("-std=c++17")
            }
        }
    }

    // 核心修复：开启统一构建系统，适配 Gradle 9.1
    experimentalProperties["android.experimental.cxx.useUnifiedBuild"] = true

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }

    kotlinOptions {
        jvmTarget = "21"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        // 请确保此版本与你的 Kotlin 插件版本匹配 (1.9.22 对应 1.5.10)
        kotlinCompilerExtensionVersion = "1.5.10"
    }

    // 【关键修复】统一打包配置块
    packaging {
        jniLibs {
            // 1. 解决 16KB Page Size 设备安装报错 (强制压缩 .so)
            // 注意：Kotlin DSL 必须带等号
            useLegacyPackaging = true

            // 2. 解决 .so 重复冲突
            pickFirsts.add("lib/arm64-v8a/libc++_shared.so")
            pickFirsts.add("lib/arm64-v8a/libMNN.so")
            pickFirsts.add("lib/arm64-v8a/libMNNOpenCV.so")
            pickFirsts.add("lib/arm64-v8a/libMNN_Express.so")
        }

        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    // 解决资源链接报错 (Theme.MaterialComponents)
    implementation("com.google.android.material:material:1.12.0")

    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.0")
    implementation("androidx.activity:activity-compose:1.10.0")
    implementation(platform("androidx.compose:compose-bom:2025.02.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
}