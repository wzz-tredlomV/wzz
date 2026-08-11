package com.openrocket.launcher.ui.viewmodel

import android.app.Application
import android.content.Context
import android.net.Uri
import android.os.StatFs
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.openrocket.launcher.engine.JavaAppManager
import com.openrocket.launcher.engine.JavaAppSetupState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import timber.log.Timber
import java.io.File

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

data class AppSettings(
    val memoryLimit: String = "512m",
    val extraJvmArgs: String = "",
    val jdkUrl: String = com.openrocket.launcher.engine.JavaAppManager.DEFAULT_JDK_URL,
    val openRocketUrl: String = com.openrocket.launcher.engine.OpenRocketManager.DEFAULT_OPENROCKET_URL,
    val autoStart: Boolean = false,
    val keepScreenOn: Boolean = true,
    val darkTheme: Boolean = true
)

sealed class JdkInstallState {
    object Idle : JdkInstallState()
    object Checking : JdkInstallState()
    data class Downloading(val progress: Float) : JdkInstallState()
    data class Extracting(val progress: Float, val currentEntry: String = "") : JdkInstallState()
    object Testing : JdkInstallState()
    data class Ready(val versionInfo: String = "", val testLog: String = "") : JdkInstallState()
    data class Error(val message: String) : JdkInstallState()
}

class SettingsViewModel(application: Application) : AndroidViewModel(application) {

    private val dataStore = application.dataStore
    private val javaAppManager = JavaAppManager(application)

    private val _settings = MutableStateFlow(AppSettings())
    val settings: StateFlow<AppSettings> = _settings.asStateFlow()

    private val _storageInfo = MutableStateFlow("")
    val storageInfo: StateFlow<String> = _storageInfo.asStateFlow()

    private val _jdkInstallState = MutableStateFlow<JdkInstallState>(JdkInstallState.Idle)
    val jdkInstallState: StateFlow<JdkInstallState> = _jdkInstallState.asStateFlow()

    private val _jdkTestLog = MutableStateFlow("")
    val jdkTestLog: StateFlow<String> = _jdkTestLog.asStateFlow()

    companion object {
        val MEMORY_LIMIT = stringPreferencesKey("memory_limit")
        val EXTRA_JVM_ARGS = stringPreferencesKey("extra_jvm_args")
        val JDK_URL = stringPreferencesKey("jdk_url")
        val OPENROCKET_URL = stringPreferencesKey("openrocket_url")
        val AUTO_START = booleanPreferencesKey("auto_start")
        val KEEP_SCREEN_ON = booleanPreferencesKey("keep_screen_on")
        val DARK_THEME = booleanPreferencesKey("dark_theme")
    }

    init {
        viewModelScope.launch {
            dataStore.data.map { prefs ->
                AppSettings(
                    memoryLimit = prefs[MEMORY_LIMIT] ?: "512m",
                    extraJvmArgs = prefs[EXTRA_JVM_ARGS] ?: "",
                    jdkUrl = prefs[JDK_URL] ?: com.openrocket.launcher.engine.JavaAppManager.DEFAULT_JDK_URL,
                    openRocketUrl = prefs[OPENROCKET_URL] ?: com.openrocket.launcher.engine.OpenRocketManager.DEFAULT_OPENROCKET_URL,
                    autoStart = prefs[AUTO_START] ?: false,
                    keepScreenOn = prefs[KEEP_SCREEN_ON] ?: true,
                    darkTheme = prefs[DARK_THEME] ?: true
                )
            }.collect { settings ->
                _settings.value = settings
            }
        }

        // Collect setup state from JavaAppManager
        viewModelScope.launch {
            javaAppManager.setupState.collect { state ->
                _jdkInstallState.value = when (state) {
                    is JavaAppSetupState.Idle -> JdkInstallState.Idle
                    is JavaAppSetupState.Checking -> JdkInstallState.Checking
                    is JavaAppSetupState.DownloadingJdk -> JdkInstallState.Downloading(state.progress)
                    is JavaAppSetupState.DownloadingApp -> JdkInstallState.Idle
                    is JavaAppSetupState.Extracting -> JdkInstallState.Extracting(state.progress, state.currentEntry)
                    is JavaAppSetupState.Testing -> JdkInstallState.Testing
                    is JavaAppSetupState.Ready -> JdkInstallState.Ready(state.versionInfo, _jdkTestLog.value)
                    is JavaAppSetupState.Error -> JdkInstallState.Error(state.message)
                }
            }
        }
    }

    fun installJdk(jdkUrl: String) {
        viewModelScope.launch {
            _jdkTestLog.value = ""
            javaAppManager.setupJdk(jdkUrl)
        }
    }

    fun installJdkFromLocalFile(uri: Uri) {
        viewModelScope.launch {
            _jdkTestLog.value = ""
            javaAppManager.installJdkFromLocalFile(uri)
        }
    }

    fun checkJdkInstalled(): Boolean {
        val context = getApplication<Application>()
        val jdkDir = File(context.filesDir, "jdk")
        val direct = File(jdkDir, "bin/java")
        if (direct.exists() && direct.length() > 0) return true
        jdkDir.listFiles()?.filter { it.isDirectory }?.forEach { subDir ->
            val nested = File(subDir, "bin/java")
            if (nested.exists() && nested.length() > 0) return true
        }
        return false
    }

    fun getJdkVersion(): String? {
        val context = getApplication<Application>()
        val jdkDir = File(context.filesDir, "jdk")

        // Find java binary
        fun findJavaBinary(): File {
            val direct = File(jdkDir, "bin/java")
            if (direct.exists()) return direct
            jdkDir.listFiles()?.filter { it.isDirectory }?.forEach { subDir ->
                val nested = File(subDir, "bin/java")
                if (nested.exists()) return nested
            }
            return direct
        }

        val javaBinary = findJavaBinary()
        val actualJdkDir = javaBinary.parentFile?.parentFile ?: jdkDir

        // Strategy 1: Read release file
        try {
            val releaseFile = File(actualJdkDir, "release")
            if (releaseFile.exists()) {
                val version = releaseFile.readText().lineSequence()
                    .map { it.trim() }
                    .find { it.startsWith("JAVA_VERSION=") }
                    ?.substringAfter("JAVA_VERSION=")
                    ?.trim('"', '\'')
                if (version != null) return version
            }
            // Check nested directory
            jdkDir.listFiles()?.filter { it.isDirectory }?.forEach { subDir ->
                val nestedRelease = File(subDir, "release")
                if (nestedRelease.exists()) {
                    val version = nestedRelease.readText().lineSequence()
                        .map { it.trim() }
                        .find { it.startsWith("JAVA_VERSION=") }
                        ?.substringAfter("JAVA_VERSION=")
                        ?.trim('"', '\'')
                    if (version != null) return version
                }
            }
        } catch (e: Exception) {
            Timber.e(e, "Failed to read release file")
        }

        // Strategy 2: Execute java -version
        return try {
            val pb = ProcessBuilder(javaBinary.absolutePath, "-version")
                .redirectErrorStream(true)
            val libDir = File(actualJdkDir, "lib")
            val libServerDir = File(libDir, "server")
            val ldPaths = mutableListOf<String>()
            if (libDir.exists()) ldPaths.add(libDir.absolutePath)
            if (libServerDir.exists()) ldPaths.add(libServerDir.absolutePath)
            if (ldPaths.isNotEmpty()) {
                pb.environment()["LD_LIBRARY_PATH"] = ldPaths.joinToString(":")
            }
            val process = pb.start()
            val reader = process.inputStream.bufferedReader()
            val output = StringBuilder()
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                output.append(line).append("\n")
            }
            process.waitFor()
            val outputStr = output.toString().trim()
            if (outputStr.isNotEmpty()) {
                val versionRegex = Regex("""version\s+\"([^\"]+)\"""")
                val match = versionRegex.find(outputStr)
                match?.groupValues?.get(1) ?: outputStr.lineSequence().firstOrNull()
            } else {
                null
            }
        } catch (e: Exception) {
            Timber.e(e, "Failed to get JDK version from java binary")
            null
        }
    }

    fun runJdkTest() {
        viewModelScope.launch {
            _jdkTestLog.value = "Running JDK tests...\n"
            try {
                val context = getApplication<Application>()
                val jdkDir = File(context.filesDir, "jdk")
                val logBuilder = StringBuilder()

                // Find java binary (handle nested directories)
                fun findJavaBinary(): File {
                    val direct = File(jdkDir, "bin/java")
                    if (direct.exists()) return direct
                    jdkDir.listFiles()?.filter { it.isDirectory }?.forEach { subDir ->
                        val nested = File(subDir, "bin/java")
                        if (nested.exists()) return nested
                    }
                    return direct
                }

                fun findBinDir(): File? {
                    val direct = File(jdkDir, "bin")
                    if (direct.exists() && direct.isDirectory) return direct
                    jdkDir.listFiles()?.filter { it.isDirectory }?.forEach { subDir ->
                        val nested = File(subDir, "bin")
                        if (nested.exists() && nested.isDirectory) return nested
                    }
                    return null
                }

                val javaBinary = findJavaBinary()
                val actualJdkDir = javaBinary.parentFile?.parentFile ?: jdkDir
                val binDir = findBinDir()
                val libDir = File(actualJdkDir, "lib")
                val libServerDir = File(libDir, "server")

                // Test 1: Check java binary exists
                logBuilder.appendLine("=== JDK Test Report ===")
                logBuilder.appendLine()

                logBuilder.appendLine("[1] Java Binary")
                logBuilder.appendLine("    Path: ${javaBinary.absolutePath}")
                logBuilder.appendLine("    Exists: ${javaBinary.exists()}")
                logBuilder.appendLine("    Size: ${javaBinary.length()} bytes")
                logBuilder.appendLine("    Executable: ${javaBinary.canExecute()}")
                logBuilder.appendLine()

                // Test 2: Check JDK directory structure
                logBuilder.appendLine("[2] JDK Directory Structure")
                logBuilder.appendLine("    JDK Home: ${actualJdkDir.absolutePath}")
                logBuilder.appendLine("    Exists: ${actualJdkDir.exists()}")
                logBuilder.appendLine("    bin/ exists: ${binDir?.exists()}")
                logBuilder.appendLine("    lib/ exists: ${libDir.exists()}")
                if (binDir != null && binDir.exists()) {
                    val binaries = binDir.listFiles()?.map { it.name }?.sorted() ?: emptyList()
                    logBuilder.appendLine("    Binaries: ${binaries.take(10).joinToString(", ")}${if (binaries.size > 10) "..." else ""}")
                }
                logBuilder.appendLine()

                // Test 3: Check release file
                val releaseFile = File(actualJdkDir, "release")
                logBuilder.appendLine("[3] Release File")
                logBuilder.appendLine("    Path: ${releaseFile.absolutePath}")
                logBuilder.appendLine("    Exists: ${releaseFile.exists()}")
                if (releaseFile.exists()) {
                    val content = releaseFile.readText().lines().take(10).joinToString("\n    ")
                    logBuilder.appendLine("    Content:\n    $content")
                }
                logBuilder.appendLine()

                // Test 4: Check LD_LIBRARY_PATH
                logBuilder.appendLine("[4] Dynamic Libraries")
                logBuilder.appendLine("    lib/ exists: ${libDir.exists()}")
                logBuilder.appendLine("    lib/server/ exists: ${libServerDir.exists()}")
                if (libServerDir.exists()) {
                    val libs = libServerDir.listFiles()?.map { it.name }?.filter { it.endsWith(".so") }?.sorted() ?: emptyList()
                    logBuilder.appendLine("    .so files: ${libs.joinToString(", ")}")
                }
                logBuilder.appendLine()

                // Test 5: Try to get version from release file
                logBuilder.appendLine("[5] Version Detection")
                val version = try {
                    releaseFile.readText().lineSequence()
                        .map { it.trim() }
                        .find { it.startsWith("JAVA_VERSION=") }
                        ?.substringAfter("JAVA_VERSION=")
                        ?.trim('"', '\'')
                } catch (e: Exception) {
                    null
                }
                logBuilder.appendLine("    From release file: ${version ?: "FAILED"}")
                logBuilder.appendLine()

                // Test 6: Try to execute java -version
                logBuilder.appendLine("[6] Java Execution Test")
                try {
                    val pb = ProcessBuilder(javaBinary.absolutePath, "-version")
                        .redirectErrorStream(true)
                    val ldPaths = mutableListOf<String>()
                    if (libDir.exists()) ldPaths.add(libDir.absolutePath)
                    if (libServerDir.exists()) ldPaths.add(libServerDir.absolutePath)
                    if (ldPaths.isNotEmpty()) {
                        pb.environment()["LD_LIBRARY_PATH"] = ldPaths.joinToString(":")
                    }
                    val process = pb.start()
                    val output = process.inputStream.bufferedReader().readText()
                    val exitCode = process.waitFor()
                    logBuilder.appendLine("    Exit Code: $exitCode")
                    logBuilder.appendLine("    Output: ${output.trim().take(200)}")
                } catch (e: Exception) {
                    logBuilder.appendLine("    FAILED: ${e.javaClass.simpleName}: ${e.message}")
                }
                logBuilder.appendLine()

                // Test 7: JDK Tools Version Info
                logBuilder.appendLine("[7] JDK Tools Version Information")
                val tools = listOf("javac", "jar", "javadoc", "jdb", "jcmd", "jconsole", "jdeps", "jfr", "jhsdb", "jimage", "jinfo", "jlink", "jmap", "jmod", "jpackage", "jps", "jrunscript", "jshell", "jstack", "jstat", "jstatd", "keytool", "rmid", "rmiregistry", "serialver")
                tools.forEach { toolName ->
                    try {
                        val toolFile = File(binDir, toolName)
                        if (toolFile.exists()) {
                            val pb = ProcessBuilder(toolFile.absolutePath, "-version")
                                .redirectErrorStream(true)
                            val ldPaths = mutableListOf<String>()
                            if (libDir.exists()) ldPaths.add(libDir.absolutePath)
                            if (libServerDir.exists()) ldPaths.add(libServerDir.absolutePath)
                            if (ldPaths.isNotEmpty()) {
                                pb.environment()["LD_LIBRARY_PATH"] = ldPaths.joinToString(":")
                            }
                            val process = pb.start()
                            val output = process.inputStream.bufferedReader().readText().trim()
                            val exitCode = process.waitFor()
                            val versionLine = output.lineSequence().firstOrNull() ?: ""
                            logBuilder.appendLine("    $toolName: ${versionLine.take(60)}")
                        } else {
                            logBuilder.appendLine("    $toolName: not found")
                        }
                    } catch (e: Exception) {
                        logBuilder.appendLine("    $toolName: error - ${e.javaClass.simpleName}")
                    }
                }
                logBuilder.appendLine()

                logBuilder.appendLine("=== Test Complete ===")

                val logText = logBuilder.toString()
                _jdkTestLog.value = logText
                Timber.i(logText)
            } catch (e: Exception) {
                _jdkTestLog.value = "Test failed: ${e.message}"
                Timber.e(e, "JDK test failed")
            }
        }
    }

    private fun updateSetting(key: Preferences.Key<String>, value: String) {
        viewModelScope.launch {
            dataStore.edit { prefs ->
                prefs[key] = value
            }
        }
    }

    private fun updateSetting(key: Preferences.Key<Boolean>, value: Boolean) {
        viewModelScope.launch {
            dataStore.edit { prefs ->
                prefs[key] = value
            }
        }
    }

    fun updateMemoryLimit(value: String) = updateSetting(MEMORY_LIMIT, value)
    fun updateExtraJvmArgs(value: String) = updateSetting(EXTRA_JVM_ARGS, value)
    fun updateJdkUrl(value: String) = updateSetting(JDK_URL, value)
    fun updateOpenRocketUrl(value: String) = updateSetting(OPENROCKET_URL, value)
    fun updateAutoStart(value: Boolean) = updateSetting(AUTO_START, value)
    fun updateKeepScreenOn(value: Boolean) = updateSetting(KEEP_SCREEN_ON, value)
    fun updateDarkTheme(value: Boolean) = updateSetting(DARK_THEME, value)

    fun clearCache() {
        viewModelScope.launch {
            try {
                val cacheDir = getApplication<Application>().cacheDir
                cacheDir.listFiles()?.forEach { file ->
                    if (file.isDirectory) {
                        file.deleteRecursively()
                    } else {
                        file.delete()
                    }
                }
                Timber.i("Cache cleared")
            } catch (e: Exception) {
                Timber.e(e, "Failed to clear cache")
            }
        }
    }

    fun resetSettings() {
        viewModelScope.launch {
            // Clear all DataStore preferences
            dataStore.edit { prefs ->
                prefs.clear()
            }
            // Reset in-memory state
            _jdkTestLog.value = ""
            _storageInfo.value = ""
            // Reset JDK install state to reflect current reality
            _jdkInstallState.value = if (checkJdkInstalled()) {
                JdkInstallState.Ready(getJdkVersion() ?: "", "")
            } else {
                JdkInstallState.Idle
            }
            Timber.i("Settings reset to defaults")
        }
    }

    fun showStorageInfo() {
        viewModelScope.launch {
            try {
                val context = getApplication<Application>()
                val filesDir = context.filesDir
                val cacheDir = context.cacheDir

                val filesSize = getDirectorySize(filesDir)
                val cacheSize = getDirectorySize(cacheDir)
                val totalAppSize = filesSize + cacheSize

                val stat = StatFs(filesDir.path)
                val availableBytes = stat.availableBytes
                val totalBytes = stat.totalBytes

                val info = buildString {
                    appendLine("App Storage Usage:")
                    appendLine("  Files: ${formatSize(filesSize)}")
                    appendLine("  Cache: ${formatSize(cacheSize)}")
                    appendLine("  Total: ${formatSize(totalAppSize)}")
                    appendLine()
                    appendLine("Device Storage:")
                    appendLine("  Available: ${formatSize(availableBytes)}")
                    appendLine("  Total: ${formatSize(totalBytes)}")
                }

                _storageInfo.value = info
                Timber.i("Storage info: $info")
            } catch (e: Exception) {
                Timber.e(e, "Failed to get storage info")
            }
        }
    }

    private fun getDirectorySize(dir: File): Long {
        var size = 0L
        dir.listFiles()?.forEach { file ->
            size += if (file.isDirectory) {
                getDirectorySize(file)
            } else {
                file.length()
            }
        }
        return size
    }

    private fun formatSize(size: Long): String {
        return when {
            size < 1024 -> "$size B"
            size < 1024 * 1024 -> "${size / 1024} KB"
            size < 1024 * 1024 * 1024 -> "${size / (1024 * 1024)} MB"
            else -> "${size / (1024 * 1024 * 1024)} GB"
        }
    }
}
