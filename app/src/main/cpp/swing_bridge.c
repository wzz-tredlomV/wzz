/**
 * JNI Bridge for Java AWT/Swing -> Android Canvas
 * 
 * This is a Phase 2 experimental module that intercepts AWT Graphics2D
 * draw calls and forwards them to Android's native Canvas/Skia via JNI.
 * 
 * Architecture:
 *   Java Swing (Graphics2D) -> JNI -> Android Canvas -> Skia (GPU)
 * 
 * Status: EXPERIMENTAL - Not yet functional
 * This requires a custom OpenJDK build with modified AWT peers.
 */

#include <jni.h>
#include <android/log.h>
#include <stdlib.h>
#include <string.h>

#define LOG_TAG "SwingBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global references
static JavaVM* g_vm = NULL;
static jobject g_surface = NULL;
static jclass g_canvasClass = NULL;
static jobject g_canvas = NULL;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    g_vm = vm;
    LOGI("SwingBridge JNI loaded");
    return JNI_VERSION_1_6;
}

/**
 * Initialize the bridge with an Android Surface
 * Called from Java when the SurfaceView is ready
 */
JNIEXPORT void JNICALL
Java_com_openrocket_launcher_engine_SwingBridge_nativeInit(JNIEnv* env, jobject thiz, jobject surface) {
    LOGI("Initializing SwingBridge with surface");
    
    if (g_surface != NULL) {
        (*env)->DeleteGlobalRef(env, g_surface);
    }
    g_surface = (*env)->NewGlobalRef(env, surface);
    
    // Get ANativeWindow from Surface
    // This would require linking against libandroid
    // For now, we just log the initialization
    LOGI("Surface stored for bridge");
}

/**
 * Draw a rectangle to the Android canvas
 * Called from modified AWT Graphics2D.drawRect()
 */
JNIEXPORT void JNICALL
Java_com_openrocket_launcher_engine_SwingBridge_nativeDrawRect(
    JNIEnv* env, jobject thiz,
    jint x, jint y, jint width, jint height,
    jint color, jint strokeWidth
) {
    LOGI("DrawRect: x=%d y=%d w=%d h=%d color=%08x", x, y, width, height, color);
    
    // In a full implementation:
    // 1. Lock the ANativeWindow
    // 2. Get the canvas
    // 3. Draw the rectangle using Skia/Canvas API
    // 4. Unlock and post
    
    // This is a stub - actual implementation requires:
    // - Custom OpenJDK build with modified X11/AWT peers
    // - ANativeWindow_lock/unlock integration
    // - Color space conversion (Java ARGB -> Android)
}

/**
 * Draw text to the Android canvas
 */
JNIEXPORT void JNICALL
Java_com_openrocket_launcher_engine_SwingBridge_nativeDrawString(
    JNIEnv* env, jobject thiz,
    jstring text, jint x, jint y,
    jint color, jfloat fontSize
) {
    const char* ctext = (*env)->GetStringUTFChars(env, text, NULL);
    LOGI("DrawString: \"%s\" at (%d,%d) color=%08x size=%f", ctext, x, y, color, fontSize);
    (*env)->ReleaseStringUTFChars(env, text, ctext);
}

/**
 * Draw an image/bitmap to the Android canvas
 */
JNIEXPORT void JNICALL
Java_com_openrocket_launcher_engine_SwingBridge_nativeDrawImage(
    JNIEnv* env, jobject thiz,
    jintArray pixels, jint width, jint height,
    jint x, jint y
) {
    LOGI("DrawImage: %dx%d at (%d,%d)", width, height, x, y);
    
    // In a full implementation:
    // 1. Convert Java int[] ARGB pixels to Android Bitmap
    // 2. Draw Bitmap to Canvas
    // 3. Clean up
}

/**
 * Flush pending draw operations to screen
 */
JNIEXPORT void JNICALL
Java_com_openrocket_launcher_engine_SwingBridge_nativeFlush(JNIEnv* env, jobject thiz) {
    LOGI("Flush called");
    // Trigger surface composition
}

/**
 * Cleanup resources
 */
JNIEXPORT void JNICALL
Java_com_openrocket_launcher_engine_SwingBridge_nativeDestroy(JNIEnv* env, jobject thiz) {
    LOGI("Destroying SwingBridge");
    
    if (g_surface != NULL) {
        (*env)->DeleteGlobalRef(env, g_surface);
        g_surface = NULL;
    }
    if (g_canvasClass != NULL) {
        (*env)->DeleteGlobalRef(env, g_canvasClass);
        g_canvasClass = NULL;
    }
    if (g_canvasClass != NULL) {
        (*env)->DeleteGlobalRef(env, g_canvasClass);
        g_canvasClass = NULL;
    }
    if (g_canvas != NULL) {
        (*env)->DeleteGlobalRef(env, g_canvas);
        g_canvas = NULL;
    }
}
