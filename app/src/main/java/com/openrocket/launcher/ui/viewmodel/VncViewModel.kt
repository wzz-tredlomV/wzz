package com.openrocket.launcher.ui.viewmodel

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.openrocket.launcher.engine.JavaAppManager
import com.openrocket.launcher.engine.ProcessState
import com.openrocket.launcher.engine.VncServerManager
import com.openrocket.launcher.vnc.VncClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import timber.log.Timber
import java.io.File

sealed class VncConnectionState {
    object Disconnected : VncConnectionState()
    object Connecting : VncConnectionState()
    object Connected : VncConnectionState()
    data class Error(val message: String) : VncConnectionState()
}

sealed class VncServerState {
    object NotRunning : VncServerState()
    object Checking : VncServerState()
    object Starting : VncServerState()
    data class Running(val display: String, val port: Int) : VncServerState()
    data class Error(val message: String) : VncServerState()
}

/**
 * ViewModel for VNC screen. Manages both the VNC server (Xvfb + x11vnc)
 * and the VNC client connection.
 */
class VncViewModel(application: Application) : AndroidViewModel(application) {

    private val javaAppManager = JavaAppManager(application)
    private val vncServerManager = VncServerManager.getInstance(application)
    private val vncClient = VncClient()

    private val _connectionState = MutableStateFlow<VncConnectionState>(VncConnectionState.Disconnected)
    val connectionState: StateFlow<VncConnectionState> = _connectionState.asStateFlow()

    private val _vncServerState = MutableStateFlow<VncServerState>(VncServerState.NotRunning)
    val vncServerState: StateFlow<VncServerState> = _vncServerState.asStateFlow()

    private val _framebuffer = MutableStateFlow<Bitmap?>(null)
    val framebuffer: StateFlow<Bitmap?> = _framebuffer.asStateFlow()

    private val _vncHost = MutableStateFlow("127.0.0.1")
    val vncHost: StateFlow<String> = _vncHost.asStateFlow()

    private val _vncPort = MutableStateFlow(5901)
    val vncPort: StateFlow<Int> = _vncPort.asStateFlow()

    private val _processState = MutableStateFlow<ProcessState>(ProcessState.Idle)
    val processState: StateFlow<ProcessState> = _processState.asStateFlow()

    private val _serverLog = MutableStateFlow("")
    val serverLog: StateFlow<String> = _serverLog.asStateFlow()

    private var javaProcess: Process? = null
    private var framebufferUpdateJob: Job? = null

    init {
        checkVncAvailability()
    }

    fun setVncHost(host: String) {
        _vncHost.value = host
    }

    fun setVncPort(port: Int) {
        _vncPort.value = port
    }

    /**
     * Check if VNC server binaries are available.
     */
    fun checkVncAvailability() {
        viewModelScope.launch {
            _vncServerState.value = VncServerState.Checking
            val available = vncServerManager.prepareBinaries()
            _vncServerState.value = if (available) {
                VncServerState.NotRunning
            } else {
                VncServerState.Error(
                    "VNC server binaries not found in assets.\n" +
                    "Please ensure assets/vnc/bin/Xvfb and assets/vnc/bin/x11vnc exist."
                )
            }
        }
    }

    /**
     * Start the built-in VNC server (Xvfb + x11vnc) and connect the client.
     * This is the main entry point for GUI mode.
     */
    fun startGuiSession(display: String = ":1", port: Int = 5901) {
        viewModelScope.launch {
            try {
                _vncServerState.value = VncServerState.Starting
                _connectionState.value = VncConnectionState.Connecting
                _serverLog.value = "Starting VNC server...\n"

                // Check JDK
                if (!javaAppManager.checkSetup()) {
                    _vncServerState.value = VncServerState.Error("JDK not installed. Please install JDK first.")
                    _connectionState.value = VncConnectionState.Error("JDK not installed")
                    return@launch
                }

                // Start VNC server
                val result = vncServerManager.startServer(display, port)

                result.onSuccess { disp ->
                    _vncServerState.value = VncServerState.Running(disp, port)
                    _serverLog.value += "VNC server started on $disp:$port\n"
                    _vncHost.value = "127.0.0.1"
                    _vncPort.value = port
                    Timber.i("VNC server running on $disp:$port")

                    // Connect VNC client
                    connectVncClient("127.0.0.1", port)
                }.onFailure { e ->
                    val msg = e.message ?: "Failed to start VNC server"
                    _vncServerState.value = VncServerState.Error(msg)
                    _connectionState.value = VncConnectionState.Error(msg)
                    _serverLog.value += "ERROR: $msg\n"
                    Timber.e(e, "Failed to start VNC server")
                }

            } catch (e: Exception) {
                val msg = e.message ?: "Unknown error starting VNC server"
                _vncServerState.value = VncServerState.Error(msg)
                _connectionState.value = VncConnectionState.Error(msg)
                _serverLog.value += "ERROR: $msg\n"
                Timber.e(e, "Exception starting VNC server")
            }
        }
    }

    /**
     * Start a Java GUI application with the built-in VNC display.
     * @param appName Name of the installed app to launch
     */
    fun startGuiApp(appName: String) {
        viewModelScope.launch {
            try {
                _connectionState.value = VncConnectionState.Connecting

                if (!javaAppManager.checkSetup()) {
                    _connectionState.value = VncConnectionState.Error("JDK not installed")
                    return@launch
                }

                // Ensure VNC server is running
                if (!vncServerManager.isRunning()) {
                    _serverLog.value += "Auto-starting VNC server...\n"
                    val result = vncServerManager.startServer(":1", 5901)
                    result.onSuccess { disp ->
                        _vncServerState.value = VncServerState.Running(disp, 5901)
                        _serverLog.value += "VNC server auto-started\n"
                    }.onFailure { e ->
                        _connectionState.value = VncConnectionState.Error("VNC server failed: ${e.message}")
                        return@launch
                    }
                }

                val jarFile = javaAppManager.getAppJar(appName)
                if (jarFile == null || !jarFile.exists()) {
                    _connectionState.value = VncConnectionState.Error("App JAR not found: $appName")
                    return@launch
                }

                val display = vncServerManager.getDisplay() ?: ":1"
                launchJavaAppWithDisplay(jarFile, display)

                // Connect VNC client to view the GUI
                val port = vncServerManager.getPort() ?: 5901
                connectVncClient("127.0.0.1", port)

            } catch (e: Exception) {
                Timber.e(e, "Failed to start GUI app")
                _connectionState.value = VncConnectionState.Error(e.message ?: "Unknown error")
            }
        }
    }

    /**
     * Launch any JAR file with GUI mode on the built-in display.
     */
    fun launchGuiAppWithJar(jarFile: File) {
        viewModelScope.launch {
            try {
                _connectionState.value = VncConnectionState.Connecting

                if (!javaAppManager.checkSetup()) {
                    _connectionState.value = VncConnectionState.Error("JDK not installed")
                    return@launch
                }

                // Ensure VNC server is running
                if (!vncServerManager.isRunning()) {
                    val result = vncServerManager.startServer(":1", 5901)
                    result.onSuccess { disp ->
                        _vncServerState.value = VncServerState.Running(disp, 5901)
                    }.onFailure { e ->
                        _connectionState.value = VncConnectionState.Error("VNC server failed: ${e.message}")
                        return@launch
                    }
                }

                val display = vncServerManager.getDisplay() ?: ":1"
                launchJavaAppWithDisplay(jarFile, display)

                val port = vncServerManager.getPort() ?: 5901
                connectVncClient("127.0.0.1", port)

            } catch (e: Exception) {
                Timber.e(e, "Failed to launch GUI app")
                _connectionState.value = VncConnectionState.Error(e.message ?: "Unknown error")
            }
        }
    }

    /**
     * Connect to an external VNC server.
     */
    fun connectToExternalVnc(host: String, port: Int) {
        viewModelScope.launch {
            _vncHost.value = host
            _vncPort.value = port
            connectVncClient(host, port)
        }
    }

    /**
     * Stop everything: Java app, VNC client, VNC server.
     */
    fun stopAll() {
        viewModelScope.launch {
            disconnect()
            stopVncServer()
            stopJavaApp()
        }
    }

    /**
     * Disconnect VNC client only.
     */
    fun disconnect() {
        framebufferUpdateJob?.cancel()
        framebufferUpdateJob = null
        vncClient.disconnect()
        _connectionState.value = VncConnectionState.Disconnected
        _framebuffer.value = null
        Timber.i("VNC client disconnected")
    }

    /**
     * Stop the built-in VNC server.
     */
    fun stopVncServer() {
        viewModelScope.launch {
            vncServerManager.stopServer()
            _vncServerState.value = VncServerState.NotRunning
            _serverLog.value += "VNC server stopped\n"
            Timber.i("VNC server stopped")
        }
    }

    /**
     * Stop the Java application process.
     */
    fun stopJavaApp() {
        viewModelScope.launch {
            try {
                javaProcess?.let { process ->
                    if (process.isAlive) {
                        process.destroy()
                        if (!process.waitFor(5, java.util.concurrent.TimeUnit.SECONDS)) {
                            process.destroyForcibly()
                        }
                    }
                }
                javaProcess = null
                _processState.value = ProcessState.Idle
                Timber.i("Java app stopped")
            } catch (e: Exception) {
                Timber.e(e, "Error stopping Java app")
            }
        }
    }

    /**
     * Send a pointer event to the VNC server.
     */
    fun sendPointerEvent(x: Int, y: Int, buttonMask: Int) {
        viewModelScope.launch {
            vncClient.sendPointerEvent(x, y, buttonMask)
        }
    }

    /**
     * Send a key event to the VNC server.
     */
    fun sendKeyEvent(keySym: Int, down: Boolean) {
        viewModelScope.launch {
            vncClient.sendKeyEvent(keySym, down)
        }
    }

    // --- Private methods ---

    private fun connectVncClient(host: String, port: Int) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                _connectionState.value = VncConnectionState.Connecting

                // Disconnect any existing connection
                vncClient.disconnect()

                // Wait a moment for server to be ready
                delay(500)

                val success = vncClient.connect(host, port)
                if (success) {
                    _connectionState.value = VncConnectionState.Connected
                    _serverLog.value += "VNC client connected to $host:$port\n"
                    Timber.i("VNC client connected to $host:$port")

                    // Start framebuffer polling
                    startFramebufferPolling()
                } else {
                    _connectionState.value = VncConnectionState.Error("Failed to connect to VNC server")
                    _serverLog.value += "ERROR: Failed to connect to VNC server\n"
                }
            } catch (e: Exception) {
                Timber.e(e, "VNC client connection failed")
                _connectionState.value = VncConnectionState.Error(e.message ?: "Connection failed")
            }
        }
    }

    private fun startFramebufferPolling() {
        framebufferUpdateJob?.cancel()
        framebufferUpdateJob = viewModelScope.launch(Dispatchers.IO) {
            while (isActive && vncClient.isConnected()) {
                try {
                    val fb = vncClient.getFramebuffer()
                    if (fb != null) {
                        _framebuffer.value = fb
                    }
                    delay(33) // ~30 FPS
                } catch (e: Exception) {
                    if (isActive) {
                        Timber.w(e, "Framebuffer polling error")
                    }
                    break
                }
            }
        }
    }

    private fun launchJavaAppWithDisplay(jarFile: File, display: String) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val jvmArgs = listOf(
                    "-Xms512m",
                    "-Xmx512m",
                    "-Djava.awt.headless=false",
                    "-Dawt.useSystemAAFontSettings=lcd",
                    "-Dswing.aatext=true"
                )

                val envVars = mutableMapOf(
                    "JAVA_HOME" to javaAppManager.javaHome,
                    "DISPLAY" to display,
                    "PATH" to "${javaAppManager.javaHome}/bin:${System.getenv("PATH") ?: ""}"
                )

                // Set library paths
                val javaHome = javaAppManager.javaHome
                val libDir = File(javaHome, "lib")
                val libServerDir = File(libDir, "server")
                val vncLibDir = File(getApplication<Application>().filesDir, "vnc_server/lib")
                val ldPaths = mutableListOf<String>()
                if (libDir.exists()) ldPaths.add(libDir.absolutePath)
                if (libServerDir.exists()) ldPaths.add(libServerDir.absolutePath)
                if (vncLibDir.exists()) ldPaths.add(vncLibDir.absolutePath)

                val systemLdPath = System.getenv("LD_LIBRARY_PATH")
                if (!systemLdPath.isNullOrBlank()) ldPaths.add(systemLdPath)

                if (ldPaths.isNotEmpty()) {
                    envVars["LD_LIBRARY_PATH"] = ldPaths.joinToString(":")
                }

                val processBuilder = ProcessBuilder(
                    javaAppManager.javaBinary.absolutePath,
                    *jvmArgs.toTypedArray(),
                    "-jar",
                    jarFile.absolutePath
                ).apply {
                    directory(javaAppManager.workingDir)
                    environment().putAll(envVars)
                    redirectErrorStream(true)
                }

                val process = processBuilder.start()
                javaProcess = process
                _processState.value = ProcessState.Running

                _serverLog.value += "Java app started on display $display\n"
                Timber.i("Java GUI app started on display $display")

                // Monitor process output
                viewModelScope.launch(Dispatchers.IO) {
                    try {
                        process.inputStream.bufferedReader().use { reader ->
                            var line: String?
                            while (reader.readLine().also { line = it } != null) {
                                line?.let {
                                    Timber.d("JavaApp: $it")
                                    _serverLog.value += "$it\n"
                                }
                            }
                        }
                        val exitCode = process.waitFor()
                        _processState.value = ProcessState.Exited(exitCode)
                        _serverLog.value += "Java app exited with code $exitCode\n"
                        javaProcess = null
                    } catch (e: Exception) {
                        if (e.message?.contains("Stream closed") != true) {
                            Timber.e(e, "Error reading Java app output")
                        }
                    }
                }

            } catch (e: Exception) {
                Timber.e(e, "Failed to launch Java app with display")
                _processState.value = ProcessState.Error(e.message ?: "Launch failed")
                _serverLog.value += "ERROR: ${e.message}\n"
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        stopAll()
    }
}
