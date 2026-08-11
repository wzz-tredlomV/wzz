#include <jni.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <android/log.h>

#define LOG_TAG "JniJvmLoader"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// JLI_Launch signature from OpenJDK's java.c
// int JLI_Launch(int argc, char ** argv,
//                int jargc, const char **jargv,
//                int appclassc, const char **appclassv,
//                const char * fullversion,
//                const char * dotversion,
//                const char * pname,
//                const char * lname,
//                jboolean javaargs,
//                jboolean cpwildcard,
//                jboolean javaw,
//                jint ergo);
typedef int (*JLI_Launch_t)(int argc, char **argv,
                            int jargc, const char **jargv,
                            int appclassc, const char **appclassv,
                            const char *fullversion,
                            const char *dotversion,
                            const char *pname,
                            const char *lname,
                            jboolean javaargs,
                            jboolean cpwildcard,
                            jboolean javaw,
                            jint ergo);

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    LOGI("JniJvmLoader JNI loaded");
    return JNI_VERSION_1_6;
}

/**
 * Build argv array for JLI_Launch from JNI parameters.
 * argv[0] = "java" (program name)
 * argv[1..n] = JVM options (-D..., -Xmx..., etc.)
 * argv[n+1] = "-jar"
 * argv[n+2] = jarPath
 * argv[n+3..] = app arguments
 */
static char **build_argv(const char *jarPath, const char *mainClass,
                         jobjectArray jvmOptions, jobjectArray appArgs,
                         JNIEnv *env, int *out_argc) {
    int jvmOptCount = jvmOptions ? (*env)->GetArrayLength(env, jvmOptions) : 0;
    int appArgCount = appArgs ? (*env)->GetArrayLength(env, appArgs) : 0;

    // argc = 1 (program name) + jvmOptCount + 2 (-jar + jarPath) + appArgCount
    int argc = 1 + jvmOptCount + 2 + appArgCount;
    char **argv = (char **)calloc(argc + 1, sizeof(char *));

    int idx = 0;
    argv[idx++] = strdup("java");

    // JVM options
    for (int i = 0; i < jvmOptCount; i++) {
        jstring optStr = (jstring)(*env)->GetObjectArrayElement(env, jvmOptions, i);
        const char *cOpt = (*env)->GetStringUTFChars(env, optStr, NULL);
        argv[idx++] = strdup(cOpt);
        (*env)->ReleaseStringUTFChars(env, optStr, cOpt);
    }

    // -jar and jar path
    argv[idx++] = strdup("-jar");
    argv[idx++] = strdup(jarPath);

    // Application arguments
    for (int i = 0; i < appArgCount; i++) {
        jstring argStr = (jstring)(*env)->GetObjectArrayElement(env, appArgs, i);
        const char *cArg = (*env)->GetStringUTFChars(env, argStr, NULL);
        argv[idx++] = strdup(cArg);
        (*env)->ReleaseStringUTFChars(env, argStr, cArg);
    }

    argv[idx] = NULL;
    *out_argc = argc;
    return argv;
}

static void free_argv(char **argv, int argc) {
    for (int i = 0; i < argc; i++) {
        free(argv[i]);
    }
    free(argv);
}

JNIEXPORT jint JNICALL
Java_com_openrocket_launcher_engine_JniJvmLoader_nativeRunJar(
    JNIEnv *env, jobject thiz,
    jstring javaHome, jstring jarPath, jstring mainClass,
    jobjectArray jvmOptions, jobjectArray appArgs)
{
    const char *cJavaHome = (*env)->GetStringUTFChars(env, javaHome, NULL);
    const char *cJarPath = (*env)->GetStringUTFChars(env, jarPath, NULL);
    const char *cMainClass = (*env)->GetStringUTFChars(env, mainClass, NULL);

    LOGI("=== JLI_Launch starting ===");
    LOGI("JAVA_HOME: %s", cJavaHome);
    LOGI("JAR: %s", cJarPath);
    LOGI("MainClass: %s", cMainClass);

    // Step 1: Build LD_LIBRARY_PATH with all JDK lib directories
    char ldLibraryPath[2048];
    snprintf(ldLibraryPath, sizeof(ldLibraryPath),
        "%s/lib:%s/lib/server:%s/lib/jli:%s/lib/client",
        cJavaHome, cJavaHome, cJavaHome, cJavaHome);

    // Also append current LD_LIBRARY_PATH if exists
    const char *currentLdPath = getenv("LD_LIBRARY_PATH");
    if (currentLdPath && strlen(currentLdPath) > 0) {
        strncat(ldLibraryPath, ":", sizeof(ldLibraryPath) - strlen(ldLibraryPath) - 1);
        strncat(ldLibraryPath, currentLdPath, sizeof(ldLibraryPath) - strlen(ldLibraryPath) - 1);
    }

    setenv("LD_LIBRARY_PATH", ldLibraryPath, 1);
    setenv("JAVA_HOME", cJavaHome, 1);
    LOGI("LD_LIBRARY_PATH=%s", ldLibraryPath);

    // Step 2: Find libjli.so (Java Launcher Interface)
    char libjliPath[512];
    snprintf(libjliPath, sizeof(libjliPath), "%s/lib/jli/libjli.so", cJavaHome);

    void *jliHandle = dlopen(libjliPath, RTLD_NOW | RTLD_GLOBAL);
    if (!jliHandle) {
        LOGE("Failed to dlopen libjli.so: %s", dlerror());
        LOGE("Tried: %s", libjliPath);

        // Try alternative paths
        const char *altPaths[] = {
            "%s/lib/libjli.so",
            "%s/jre/lib/jli/libjli.so",
            "%s/jre/lib/libjli.so"
        };
        for (int i = 0; i < 4; i++) {
            snprintf(libjliPath, sizeof(libjliPath), altPaths[i], cJavaHome);
            jliHandle = dlopen(libjliPath, RTLD_NOW | RTLD_GLOBAL);
            if (jliHandle) {
                LOGI("Found libjli.so at alternative path: %s", libjliPath);
                break;
            }
        }

        if (!jliHandle) {
            LOGE("Could not find libjli.so anywhere in %s", cJavaHome);
            (*env)->ReleaseStringUTFChars(env, javaHome, cJavaHome);
            (*env)->ReleaseStringUTFChars(env, jarPath, cJarPath);
            (*env)->ReleaseStringUTFChars(env, mainClass, cMainClass);
            return -1;
        }
    }

    LOGI("libjli.so loaded from: %s", libjliPath);

    // Step 3: Get JLI_Launch function pointer
    JLI_Launch_t JLI_Launch = (JLI_Launch_t)dlsym(jliHandle, "JLI_Launch");
    if (!JLI_Launch) {
        LOGE("Failed to find JLI_Launch: %s", dlerror());
        dlclose(jliHandle);
        (*env)->ReleaseStringUTFChars(env, javaHome, cJavaHome);
        (*env)->ReleaseStringUTFChars(env, jarPath, cJarPath);
        (*env)->ReleaseStringUTFChars(env, mainClass, cMainClass);
        return -2;
    }

    LOGI("JLI_Launch function resolved");

    // Step 4: Build argv for JLI_Launch
    int argc = 0;
    char **argv = build_argv(cJarPath, cMainClass, jvmOptions, appArgs, env, &argc);

    LOGI("JLI_Launch argc=%d", argc);
    for (int i = 0; i < argc; i++) {
        LOGI("  argv[%d] = %s", i, argv[i]);
    }

    // Step 5: Call JLI_Launch
    // This function will:
    // 1. Parse JVM options
    // 2. Load libjvm.so (via dlopen internally)
    // 3. Create JVM via JNI_CreateJavaVM
    // 4. Find main class and invoke main()
    // 5. All WITHOUT calling exec()!
    LOGI("Calling JLI_Launch...");
    int exitCode = JLI_Launch(
        argc, argv,
        0, NULL,           // jargc, jargv (not used)
        0, NULL,           // appclassc, appclassv (not used)
        "",                // fullversion
        "",                // dotversion
        "java",            // pname
        "java",            // lname
        JNI_FALSE,         // javaargs
        JNI_TRUE,          // cpwildcard
        JNI_FALSE,         // javaw
        0                  // ergo
    );

    LOGI("JLI_Launch returned: %d", exitCode);

    // Cleanup
    free_argv(argv, argc);
    dlclose(jliHandle);

    (*env)->ReleaseStringUTFChars(env, javaHome, cJavaHome);
    (*env)->ReleaseStringUTFChars(env, jarPath, cJarPath);
    (*env)->ReleaseStringUTFChars(env, mainClass, cMainClass);

    LOGI("=== JLI_Launch completed ===");
    return exitCode;
}
