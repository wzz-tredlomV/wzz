package com.openrocket.launcher.engine

import timber.log.Timber
import java.io.File

/**
 * JNI-based JVM loader for Android 10+.
 * Loads libjvm.so via dlopen and creates an in-process JVM instance,
 * bypassing Android 10+ exec() restrictions on app_data_file.
 */
class JniJvmLoader {

    init {
        try {
            System.loadLibrary("jni_jvm_loader")
            Timber.i("jni_jvm_loader native library loaded")
        } catch (e: UnsatisfiedLinkError) {
            Timber.e(e, "Failed to load jni_jvm_loader native library")
            throw e
        }
    }

    /**
     * Run a JAR file by creating an in-process JVM via JNI.
     *
     * @param javaHome Path to JDK home directory
     * @param jarPath Path to JAR file
     * @param mainClass Fully qualified main class name (e.g., "com.example.Main")
     * @param jvmOptions JVM options like "-Xmx256m"
     * @param appArgs Application arguments passed to main()
     * @return Exit code (0 for success, non-zero for failure)
     */
    fun runJar(
        javaHome: String,
        jarPath: String,
        mainClass: String,
        jvmOptions: Array<String> = emptyArray(),
        appArgs: Array<String> = emptyArray()
    ): Int {
        Timber.i("JNI runJar: javaHome=$javaHome, jar=$jarPath, mainClass=$mainClass")

        val libJvmPaths = listOf(
            "$javaHome/lib/server/libjvm.so",
            "$javaHome/lib/client/libjvm.so"
        )
        val libJvmExists = libJvmPaths.any { File(it).exists() }
        if (!libJvmExists) {
            Timber.e("libjvm.so not found in $javaHome")
            return -1
        }

        return try {
            nativeRunJar(javaHome, jarPath, mainClass, jvmOptions, appArgs)
        } catch (e: Exception) {
            Timber.e(e, "JNI runJar failed")
            -1
        }
    }

    private external fun nativeRunJar(
        javaHome: String,
        jarPath: String,
        mainClass: String,
        jvmOptions: Array<String>,
        appArgs: Array<String>
    ): Int
}
