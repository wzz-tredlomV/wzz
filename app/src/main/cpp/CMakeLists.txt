cmake_minimum_required(VERSION 3.22.1)
project("java_launcher")

# JNI JVM Loader - loads libjvm.so via dlopen for Android 10+
add_library(
    jni_jvm_loader
    SHARED
    jni_jvm_loader.c
)

# Swing bridge (experimental)
add_library(
    swing_bridge
    SHARED
    swing_bridge.c
    vnc_client.c
)

find_library(
    log-lib
    log
)

target_link_libraries(
    jni_jvm_loader
    ${log-lib}
    dl
)

target_link_libraries(
    swing_bridge
    ${log-lib}
    android
)

set_target_properties(jni_jvm_loader PROPERTIES
    C_STANDARD 11
)

set_target_properties(swing_bridge PROPERTIES
    C_STANDARD 11
    CXX_STANDARD 17
)
