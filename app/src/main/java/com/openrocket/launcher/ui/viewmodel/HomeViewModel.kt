package com.openrocket.launcher.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.openrocket.launcher.engine.JavaAppManager
import com.openrocket.launcher.engine.JavaAppSetupState
import com.openrocket.launcher.engine.JarExecutor
import com.openrocket.launcher.engine.ProcessState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import timber.log.Timber
import java.io.File

class HomeViewModel(application: Application) : AndroidViewModel(application) {

    private val javaAppManager = JavaAppManager(application)
    private val jarExecutor = JarExecutor.getInstance(application)

    private val _setupState = MutableStateFlow<JavaAppSetupState>(JavaAppSetupState.Idle)
    val setupState: StateFlow<JavaAppSetupState> = _setupState.asStateFlow()

    private val _processState = MutableStateFlow<ProcessState>(ProcessState.Idle)
    val processState: StateFlow<ProcessState> = _processState.asStateFlow()

    private val _jdkVersion = MutableStateFlow<String>("Unknown")
    val jdkVersion: StateFlow<String> = _jdkVersion.asStateFlow()

    private val _installedApps = MutableStateFlow<List<com.openrocket.launcher.engine.InstalledApp>>(emptyList())
    val installedApps: StateFlow<List<com.openrocket.launcher.engine.InstalledApp>> = _installedApps.asStateFlow()

    private val _isRunning = MutableStateFlow(false)
    val isRunning: StateFlow<Boolean> = _isRunning.asStateFlow()

    private val _selectedApp = MutableStateFlow<String?>(null)
    val selectedApp: StateFlow<String?> = _selectedApp.asStateFlow()

    init {
        refreshInstalledApps()
        // Collect setup state once in init to avoid memory leaks from repeated checkSetup() calls
        viewModelScope.launch {
            javaAppManager.setupState.collect { state ->
                _setupState.value = state
            }
        }
    }

    fun checkSetup() {
        viewModelScope.launch {
            val ready = javaAppManager.checkSetup()
            if (ready) {
                _jdkVersion.value = javaAppManager.getJdkVersion() ?: "Unknown"
            }
        }
    }

    fun refreshInstalledApps() {
        _installedApps.value = javaAppManager.getInstalledApps()
    }

    fun selectApp(appName: String) {
        _selectedApp.value = appName
    }

    fun startJavaApp(appName: String? = _selectedApp.value, useGuiMode: Boolean = false) {
        val targetApp = appName ?: return
        val jarFile = javaAppManager.getAppJar(targetApp) ?: run {
            _processState.value = ProcessState.Error("JAR file not found for $targetApp")
            return
        }

        viewModelScope.launch {
            try {
                if (!javaAppManager.checkSetup()) {
                    _processState.value = ProcessState.Error("JDK not installed. Please run setup first.")
                    return@launch
                }

                _processState.value = ProcessState.Starting

                // Build JVM args based on mode
                val jvmArgs = mutableListOf(
                    "-Xms32m",
                    "-Xmx256m",
                    "-XX:+UseSerialGC"
                )

                if (useGuiMode) {
                    // GUI mode: set DISPLAY for X11
                    jvmArgs.add("-Djava.awt.headless=false")
                    // Note: For GUI apps, user needs VNC server running
                } else {
                    // Headless mode: for command-line tools
                    jvmArgs.add("-Djava.awt.headless=true")
                }

                val envVars = mutableMapOf(
                    "JAVA_HOME" to javaAppManager.javaHome,
                    "PATH" to "${javaAppManager.javaHome}/bin:${System.getenv("PATH") ?: ""}"
                )

                // Android 10+ extended library path
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                    val javaHome = javaAppManager.javaHome
                    val libDir = "$javaHome/lib"
                    val libServerDir = "$libDir/server"
                    val extendedLdPath = buildString {
                        append("$libDir:$libServerDir")
                        listOf("jli", "server", "client", "jvm").forEach { subdir ->
                            val subDirPath = "$libDir/$subdir"
                            if (java.io.File(subDirPath).exists()) {
                                append(":$subDirPath")
                            }
                        }
                    }
                    envVars["LD_LIBRARY_PATH"] = extendedLdPath
                }

                if (useGuiMode) {
                    envVars["DISPLAY"] = ":0"
                }

                val result = jarExecutor.startJar(
                    javaBinary = javaAppManager.javaBinary,
                    jarFile = jarFile,
                    workingDir = javaAppManager.workingDir,
                    jvmArgs = jvmArgs,
                    envVars = envVars
                )

                result.fold(
                    onSuccess = { process ->
                        _processState.value = ProcessState.Running
                        _isRunning.value = true

                        // Monitor process exit
                        viewModelScope.launch {
                            process.waitFor()
                            val exitCode = process.exitValue()
                            _processState.value = ProcessState.Exited(exitCode)
                            _isRunning.value = false
                        }
                    },
                    onFailure = { error ->
                        val errorMsg = error.message ?: "Failed to start"
                        // Provide Android 10+ specific guidance
                        val enhancedMsg = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q &&
                            errorMsg.contains("execution strategies failed", ignoreCase = true)) {
                            "$errorMsg\n\nTip: On Android 10+, you may need to:\n" +
                            "1. Use a rooted device with SELinux permissive mode\n" +
                            "2. Use a device with Android 9 or below"
                        } else {
                            errorMsg
                        }
                        _processState.value = ProcessState.Error(enhancedMsg)
                        _isRunning.value = false
                    }
                )
            } catch (e: Exception) {
                Timber.e(e, "Failed to start Java app")
                _processState.value = ProcessState.Error(e.message ?: "Failed to start Java app")
                _isRunning.value = false
            }
        }
    }

    fun stopJavaApp() {
        viewModelScope.launch {
            _processState.value = ProcessState.Stopping
            jarExecutor.stopProcess()
            _processState.value = ProcessState.Idle
            _isRunning.value = false
        }
    }

    fun setupJdk() {
        viewModelScope.launch {
            javaAppManager.setupJdk()
        }
    }
}
