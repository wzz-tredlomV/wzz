/**
 * Lightweight VNC Client for Android
 * 
 * Phase 2 GUI Rendering Solution:
 * Uses Android ANativeWindow + GPU texture for VNC framebuffer display.
 * Avoids CPU soft-rendering bottleneck by using GPU composition.
 * 
 * Protocol: RFB 3.3/3.7/3.8
 * Encoding: Raw, CopyRect, RRE, Hextile, ZRLE, Tight
 * 
 * Status: FRAMEWORK - Core structure implemented, protocol parsing stubbed
 */

#include <jni.h>
#include <android/log.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>

#define LOG_TAG "VncClient"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// RFB Protocol constants
#define RFB_VERSION_3_3 "RFB 003.003\n"
#define RFB_VERSION_3_7 "RFB 003.007\n"
#define RFB_VERSION_3_8 "RFB 003.008\n"

#define SECURITY_NONE 1
#define SECURITY_VNC_AUTH 2

#define ENCODING_RAW 0
#define ENCODING_COPYRECT 1
#define ENCODING_RRE 2
#define ENCODING_HEXTILE 5
#define ENCODING_ZRLE 16
#define ENCODING_TIGHT 7

// Client state
typedef struct {
    int socket_fd;
    int width;
    int height;
    int bpp;
    int depth;
    int big_endian;
    int true_color;
    int red_max;
    int green_max;
    int blue_max;
    int red_shift;
    int green_shift;
    int blue_shift;
    
    ANativeWindow* window;
    pthread_t thread;
    int running;
    pthread_mutex_t lock;
} VncClient;

static VncClient* g_client = NULL;

/**
 * Connect to VNC server
 */
static int vnc_connect(const char* host, int port) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        LOGE("Failed to create socket: %s", strerror(errno));
        return -1;
    }
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, host, &addr.sin_addr);
    
    if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOGE("Failed to connect to %s:%d: %s", host, port, strerror(errno));
        close(sock);
        return -1;
    }
    
    LOGI("Connected to VNC server %s:%d", host, port);
    return sock;
}

/**
 * Read exact number of bytes
 */
static int read_exact(int fd, void* buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = read(fd, (char*)buf + total, len - total);
        if (n <= 0) return -1;
        total += n;
    }
    return 0;
}

/**
 * Perform RFB handshake
 */
static int vnc_handshake(VncClient* client) {
    char version[13];
    if (read_exact(client->socket_fd, version, 12) < 0) {
        LOGE("Failed to read version");
        return -1;
    }
    version[12] = '\0';
    LOGI("Server version: %s", version);
    
    // Send our version (3.8)
    if (write(client->socket_fd, RFB_VERSION_3_8, 12) != 12) {
        LOGE("Failed to send version");
        return -1;
    }
    
    // Read security types
    uint8_t num_types;
    if (read_exact(client->socket_fd, &num_types, 1) < 0) {
        LOGE("Failed to read security types count");
        return -1;
    }
    
    if (num_types == 0) {
        // Connection failed, read reason
        uint32_t reason_len;
        read_exact(client->socket_fd, &reason_len, 4);
        reason_len = ntohl(reason_len);
        char* reason = malloc(reason_len + 1);
        read_exact(client->socket_fd, reason, reason_len);
        reason[reason_len] = '\0';
        LOGE("Connection failed: %s", reason);
        free(reason);
        return -1;
    }
    
    uint8_t types[256];
    read_exact(client->socket_fd, types, num_types);
    
    // Select None security if available
    int selected = -1;
    for (int i = 0; i < num_types; i++) {
        if (types[i] == SECURITY_NONE) {
            selected = SECURITY_NONE;
            break;
        }
    }
    
    if (selected < 0) {
        LOGE("No supported security type found");
        return -1;
    }
    
    uint8_t sel = selected;
    write(client->socket_fd, &sel, 1);
    
    // For None security, server sends 4 bytes result (0 = OK)
    if (selected == SECURITY_NONE) {
        uint32_t result;
        read_exact(client->socket_fd, &result, 4);
        result = ntohl(result);
        if (result != 0) {
            LOGE("Security handshake failed: %u", result);
            return -1;
        }
    }
    
    LOGI("Security handshake complete");
    return 0;
}

/**
 * Initialize VNC client
 */
JNIEXPORT jlong JNICALL
Java_com_openrocket_launcher_vnc_VncClient_nativeInit(
    JNIEnv* env, jobject thiz,
    jstring host, jint port, jobject surface
) {
    const char* chost = (*env)->GetStringUTFChars(env, host, NULL);
    
    VncClient* client = calloc(1, sizeof(VncClient));
    if (!client) {
        LOGE("Failed to allocate VncClient");
        (*env)->ReleaseStringUTFChars(env, host, chost);
        return 0;
    }
    
    pthread_mutex_init(&client->lock, NULL);
    client->window = ANativeWindow_fromSurface(env, surface);
    
    client->socket_fd = vnc_connect(chost, port);
    (*env)->ReleaseStringUTFChars(env, host, chost);
    
    if (client->socket_fd < 0) {
        free(client);
        return 0;
    }
    
    if (vnc_handshake(client) < 0) {
        close(client->socket_fd);
        free(client);
        return 0;
    }
    
    g_client = client;
    LOGI("VNC client initialized");
    return (jlong)client;
}

/**
 * Send framebuffer update request
 */
JNIEXPORT void JNICALL
Java_com_openrocket_launcher_vnc_VncClient_nativeRequestUpdate(
    JNIEnv* env, jobject thiz, jlong handle,
    jint x, jint y, jint width, jint height, jboolean incremental
) {
    VncClient* client = (VncClient*)handle;
    if (!client) return;
    
    uint8_t msg[10];
    msg[0] = 3; // FramebufferUpdateRequest
    msg[1] = incremental ? 1 : 0;
    *(uint16_t*)(msg + 2) = htons(x);
    *(uint16_t*)(msg + 4) = htons(y);
    *(uint16_t*)(msg + 6) = htons(width);
    *(uint16_t*)(msg + 8) = htons(height);
    
    write(client->socket_fd, msg, 10);
}

/**
 * Send pointer event
 */
JNIEXPORT void JNICALL
Java_com_openrocket_launcher_vnc_VncClient_nativeSendPointerEvent(
    JNIEnv* env, jobject thiz, jlong handle,
    jint x, jint y, jint buttonMask
) {
    VncClient* client = (VncClient*)handle;
    if (!client) return;
    
    uint8_t msg[6];
    msg[0] = 5; // PointerEvent
    msg[1] = buttonMask;
    *(uint16_t*)(msg + 2) = htons(x);
    *(uint16_t*)(msg + 4) = htons(y);
    
    write(client->socket_fd, msg, 6);
}

/**
 * Send key event
 */
JNIEXPORT void JNICALL
Java_com_openrocket_launcher_vnc_VncClient_nativeSendKeyEvent(
    JNIEnv* env, jobject thiz, jlong handle,
    jint key, jboolean down
) {
    VncClient* client = (VncClient*)handle;
    if (!client) return;
    
    uint8_t msg[8];
    msg[0] = 4; // KeyEvent
    msg[1] = down ? 1 : 0;
    *(uint16_t*)(msg + 2) = 0;
    *(uint32_t*)(msg + 4) = htonl(key);
    
    write(client->socket_fd, msg, 8);
}

/**
 * Cleanup VNC client
 */
JNIEXPORT void JNICALL
Java_com_openrocket_launcher_vnc_VncClient_nativeDestroy(
    JNIEnv* env, jobject thiz, jlong handle
) {
    VncClient* client = (VncClient*)handle;
    if (!client) return;
    
    client->running = 0;
    
    if (client->socket_fd >= 0) {
        close(client->socket_fd);
    }
    
    if (client->window) {
        ANativeWindow_release(client->window);
    }
    
    pthread_mutex_destroy(&client->lock);
    free(client);
    g_client = NULL;
    
    LOGI("VNC client destroyed");
}
