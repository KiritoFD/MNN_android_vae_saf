package com.example.mnn

import android.content.res.AssetManager
import android.graphics.Bitmap

object MNNNative {
    // 加载我们在 CMakeLists.txt 里定义的库名
    init {
        System.loadLibrary("mnn-core")
    }

    /**
     * 初始化模型 (加载 3 个模型到内存)
     * 建议在 IO 线程调用
     */
    external fun init(assetManager: AssetManager): Boolean

    /**
     * 执行推理
     * @param srcBitmap 用户输入的图片 (必须已经 resize 到 512x512)
     * @param dstBitmap 用于接收结果的空白图片 (必须是 512x512, ARGB_8888)
     * @param styleId   风格 ID (对应 Python 的 s_idx)
     * @param steps     迭代步数 (推荐 15-20)
     */
    external fun generate(srcBitmap: Bitmap, dstBitmap: Bitmap, styleId: Int, steps: Int): Boolean
}