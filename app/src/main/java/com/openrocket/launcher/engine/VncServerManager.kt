package com.openrocket.launcher.engine

import android.content.Context
import android.os.Build
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

/**
 * VNC Server state machine
 */
sealed class VncServerStatus {
    object Idle : VncServerStatus()
    object Checking : VncServerStatus()
    data class Downloading(val component: String, val progress: Float) : VncServerStatus()
    data class Installing(val message: String) : VncServerStatus()
    object Starting : VncServerStatus()
    data class Running(val display: String, val port: Int, val pid: Int?) : VncServerStatus()
    object Stopping : VncServerStatus()
    data class Error(val message: String) : VncServerStatus()
}

/**
 * Configuration for the built-in VNC server
 */
data class VncServerConfig(
    val display: String = ":0",
    val port: Int = 5900,
    val password: String = "",
    val screenResolution: String = "1280x720x24",
    val useBuiltInServer: Boolean = true,
    val externalHost: String = "localhost",
    val externalPort: Int = 5900
)

/**
 * Built-in VNC server manager with auto-download, assets bundling, and Termux fallback.
 *
 * Strategy priority:
 * 1. Native library dir (Android 10+ exec workaround — libxvfb.so / libx11vnc.so)
 * 2. APK assets (assets/vnc/bin/)
 * 3. Previously downloaded binaries (files/vnc/bin/)
 * 4. Termux system paths
 * 5. Auto-download from configurable URL
 */
class VncServerManager private constructor(private val context: Context) {

    companion object {
        @Volatile
        private var instance: VncServerManager? = null

        fun getInstance(context: Context): VncServerManager {
            return instance ?: synchronized(this) {
                instance ?: VncServerManager(context.applicationContext).also { instance = it }
            }
        }

        /** Default URL for prebuilt Android ARM64 VNC binaries */
        const val DEFAULT_DOWNLOAD_URL =
            "https://github.com/termux/termux-packages/releases/download/bootstrap/bootstrap-aarch64.zip"
    }

    private val downloadManager = DownloadManager.getInstance(context)

    private val _serverStatus = MutableStateFlow<VncServerStatus>(VncServerStatus.Idle)
    val serverStatus: StateFlow<VncServerStatus> = _serverStatus.asStateFlow()

    private var xvfbProcess: Process? = null
    private var x11vncProcess: Process? = null

    /** Private app files directory for VNC binaries */
    private val vncDir: File
        get() = File(context.filesDir, "vnc").apply { mkdirs() }

    private val binDir: File
        get() = File(vncDir, "bin").apply { mkdirs() }

    /** Native library directory — the ONLY place Android 10+ allows exec() */
    private val nativeLibDir: File
        get() = File(context.applicationInfo.nativeLibraryDir)

    private val xvfbFile: File
        get() = File(binDir, "Xvfb")

    private val x11vncFile: File
        get() = File(binDir, "x11vnc")

    // Termux fallback paths
    private val termuxXvfb = File("/data/data/com.termux/files/usr/bin/Xvfb")
    private val termuxX11vnc = File("/data/data/com.termux/files/usr/bin/x11vnc")

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /**
     * Check whether built-in VNC binaries are available anywhere.
     * Updates [_serverStatus] to [VncServerStatus.Checking] while working.
     */
    suspend fun checkBuiltInBinaries(): Boolean = withContext(Dispatchers.IO) {
        _serverStatus.value = VncServerStatus.Checking

        // 1. Native library dir (Android 10+ primary)
        if (hasNativeLibBinaries()) {
            Timber.i("VNC binaries found in nativeLibraryDir")
            _serverStatus.value = VncServerStatus.Idle
            return@withContext true
        }

        // 2. APK assets
        if (checkAssetsBinaries()) {
            copyAssetsToPrivateDir()
            makeExecutable(xvfbFile, x11vncFile)
            if (xvfbFile.exists() && x11vncFile.exists()) {
                Timber.i("VNC binaries extracted from assets")
                _serverStatus.value = VncServerStatus.Idle
                return@withContext true
            }
        }

        // 3. Previously downloaded / extracted
        if (xvfbFile.exists() && x11vncFile.exists()) {
            makeExecutable(xvfbFile, x11vncFile)
            Timber.i("VNC binaries found in private bin dir")
            _serverStatus.value = VncServerStatus.Idle
            return@withContext true
        }

        // 4. Termux
        if (termuxXvfb.exists() && termuxX11vnc.exists()) {
            Timber.i("VNC binaries found in Termux")
            _serverStatus.value = VncServerStatus.Idle
            return@withContext true
        }

        Timber.w("No VNC binaries available")
        _serverStatus.value = VncServerStatus.Error(
            "VNC server binaries not available.\n" +
            "Please install Termux and run: pkg install x11vnc xvfb\n" +
            "Or download prebuilt binaries."
        )
        false
    }

    /**
     * Download prebuilt VNC binaries from [url] and extract them.
     */
    suspend fun downloadBinaries(url: String = DEFAULT_DOWNLOAD_URL): Boolean =
        withContext(Dispatchers.IO) {
            try {
                _serverStatus.value = VncServerStatus.Downloading("VNC Server", 0f)

                val zipFile = File(context.cacheDir, "vnc_binaries.zip")
                if (zipFile.exists()) zipFile.delete()

                val result = downloadManager.download(url, zipFile, "vnc_download")
                result.fold(
                    onSuccess = { file ->
                        _serverStatus.value = VncServerStatus.Installing("Extracting binaries...")
                        extractZip(file, binDir)
                        file.delete()
                        makeExecutable(xvfbFile, x11vncFile)

                        // For Android 10+, also copy to nativeLibDir
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                            copyToNativeLibDir(xvfbFile, "libxvfb.so")
                            copyToNativeLibDir(x11vncFile, "libx11vnc.so")
                        }

                        _serverStatus.value = VncServerStatus.Idle
                        Timber.i("VNC binaries downloaded and installed")
                        true
                    },
                    onFailure = { error ->
                        _serverStatus.value = VncServerStatus.Error(
                            "Download failed: ${error.message}"
                        )
                        false
                    }
                )
            } catch (e: Exception) {
                Timber.e(e, "Failed to download VNC binaries")
                _serverStatus.value = VncServerStatus.Error("Download failed: ${e.message}")
                false
            }
        }

    /**
     * Start the built-in VNC server (Xvfb + x11vnc).
     *
     * @return true if both processes started successfully
     */
    suspend fun startBuiltInServer(config: VncServerConfig = VncServerConfig()): Boolean =
        withContext(Dispatchers.IO) {
            try {
                _serverStatus.value = VncServerStatus.Starting

                // Stop any existing server first
                stopServerInternal()

                val xvfbPath = findXvfbPath()
                val x11vncPath = findX11vncPath()

                if (xvfbPath == null || x11vncPath == null) {
                    _serverStatus.value = VncServerStatus.Error(
                        "VNC server binaries not available.\n" +
                        "Please install Termux and run: pkg install x11vnc xvfb\n" +
                        "Or download prebuilt binaries."
                    )
                    return@withContext false
                }

                // ---- Start Xvfb ----
                val xvfbCmd = mutableListOf(
                    xvfbPath,
                    config.display,
                    "-screen", "0", config.screenResolution,
                    "-ac",
                    "+extension", "GLX",
                    "+render",
                    "-noreset"
                )

                val xvfbPb = ProcessBuilder(wrapForExec(xvfbCmd))
                    .apply {
                        environment().putAll(System.getenv())
                        redirectErrorStream(true)
                    }

                xvfbProcess = xvfbPb.start()
                delay(1500)

                if (xvfbProcess?.isAlive != true) {
                    val error = readProcessError(xvfbProcess)
                    _serverStatus.value = VncServerStatus.Error("Xvfb failed: $error")
                    return@withContext false
                }
                Timber.i("Xvfb started on ${config.display}")

                // ---- Start x11vnc ----
                val x11vncCmd = mutableListOf(
                    x11vncPath,
                    "-display", config.display,
                    "-rfbport", config.port.toString(),
                    "-forever",
                    "-shared",
                    "-nopw",
                    "-xkb"
                )
                if (config.password.isNotBlank()) {
                    x11vncCmd.add("-passwd")
                    x11vncCmd.add(config.password)
                }

                val x11vncPb = ProcessBuilder(wrapForExec(x11vncCmd))
                    .apply {
                        environment().putAll(System.getenv())
                        environment()["DISPLAY"] = config.display
                        redirectErrorStream(true)
                    }

                x11vncProcess = x11vncPb.start()
                delay(1000)

                if (x11vncProcess?.isAlive == true) {
                    val pid: Int? = null // Android Process does not support pid()
                    _serverStatus.value = VncServerStatus.Running(
                        display = config.display,
                        port = config.port,
                        pid = pid
                    )
                    Timber.i("x11vnc started on port ${config.port}")
                    true
                } else {
                    val error = readProcessError(x11vncProcess)
                    stopServerInternal()
                    _serverStatus.value = VncServerStatus.Error("x11vnc failed: $error")
                    false
                }
            } catch (e: Exception) {
                Timber.e(e, "Failed to start VNC server")
                stopServerInternal()
                _serverStatus.value = VncServerStatus.Error("Start failed: ${e.message}")
                false
            }
        }

    /**
     * Start the built-in VNC server with display and port parameters.
     * @return Result containing the display string on success
     */
    suspend fun startServer(display: String, port: Int): Result<String> =
        withContext(Dispatchers.IO) {
            val config = VncServerConfig(display = display, port = port)
            val success = startBuiltInServer(config)
            if (success) {
                Result.success(display)
            } else {
                Result.failure(Exception("Failed to start VNC server on $display:$port"))
            }
        }

    /**
     * Stop the built-in VNC server.
     */
    suspend fun stopServer() = stopServerInternal()

    /**
     * Check if binaries are available (alias for checkBuiltInBinaries).
     */
    suspend fun prepareBinaries(): Boolean = checkBuiltInBinaries()

    /**
     * Check if the VNC server is currently running.
     */
    fun isRunning(): Boolean {
        return xvfbProcess?.isAlive == true && x11vncProcess?.isAlive == true
    }

    /**
     * Get the current display string if running.
     */
    fun getDisplay(): String? {
        return if (isRunning()) ":0" else null
    }

    /**
     * Get the current VNC port if running.
     */
    fun getPort(): Int? {
        return if (isRunning()) 5900 else null
    }

    /**
     * Test TCP connectivity to an external VNC server.
     */
    suspend fun testExternalConnection(host: String, port: Int): Boolean =
        withContext(Dispatchers.IO) {
            try {
                java.net.Socket(host, port).use { it.isConnected }
            } catch (_: Exception) {
                false
            }
        }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    private suspend fun stopServerInternal() = withContext(Dispatchers.IO) {
        _serverStatus.value = VncServerStatus.Stopping

        x11vncProcess?.let { proc ->
            if (proc.isAlive) {
                proc.destroy()
                try {
                    if (!proc.waitFor(3, TimeUnit.SECONDS)) {
                        proc.destroyForcibly()
                    }
                } catch (_: Exception) {
                    proc.destroyForcibly()
                }
            }
        }
        x11vncProcess = null

        xvfbProcess?.let { proc ->
            if (proc.isAlive) {
                proc.destroy()
                try {
                    if (!proc.waitFor(3, TimeUnit.SECONDS)) {
                        proc.destroyForcibly()
                    }
                } catch (_: Exception) {
                    proc.destroyForcibly()
                }
            }
        }
        xvfbProcess = null

        _serverStatus.value = VncServerStatus.Idle
        Timber.i("VNC server stopped")
    }

    /** Find Xvfb binary across all possible locations. */
    private fun findXvfbPath(): String? {
        File(nativeLibDir, "libxvfb.so").takeIf { it.exists() }?.let { return it.absolutePath }
        xvfbFile.takeIf { it.exists() }?.let { return it.absolutePath }
        termuxXvfb.takeIf { it.exists() }?.let { return it.absolutePath }
        return null
    }

    /** Find x11vnc binary across all possible locations. */
    private fun findX11vncPath(): String? {
        File(nativeLibDir, "libx11vnc.so").takeIf { it.exists() }?.let { return it.absolutePath }
        x11vncFile.takeIf { it.exists() }?.let { return it.absolutePath }
        termuxX11vnc.takeIf { it.exists() }?.let { return it.absolutePath }
        return null
    }

    /** Check if native library dir already has our .so-wrapped binaries. */
    private fun hasNativeLibBinaries(): Boolean {
        return File(nativeLibDir, "libxvfb.so").exists() &&
               File(nativeLibDir, "libx11vnc.so").exists()
    }

    /** Check APK assets for vnc/bin/ contents. */
    private fun checkAssetsBinaries(): Boolean {
        return try {
            context.assets.list("vnc/bin")?.let { list ->
                list.contains("Xvfb") && list.contains("x11vnc")
            } ?: false
        } catch (_: Exception) {
            false
        }
    }

    /** Copy Xvfb + x11vnc from assets to private bin dir. */
    private fun copyAssetsToPrivateDir() {
        arrayOf("Xvfb", "x11vnc").forEach { name ->
            try {
                val outFile = File(binDir, name)
                if (!outFile.exists()) {
                    context.assets.open("vnc/bin/$name").use { input ->
                        outFile.outputStream().use { output -> input.copyTo(output) }
                    }
                }
            } catch (e: Exception) {
                Timber.e(e, "Failed to copy $name from assets")
            }
        }
    }

    /** Extract a ZIP file to a destination directory. */
    private fun extractZip(zipFile: File, destDir: File) {
        java.util.zip.ZipInputStream(zipFile.inputStream()).use { zis ->
            var entry: java.util.zip.ZipEntry?
            while (zis.nextEntry.also { entry = it } != null) {
                val ze = entry ?: continue
                val outFile = File(destDir, ze.name)
                if (ze.isDirectory) {
                    outFile.mkdirs()
                } else {
                    outFile.parentFile?.mkdirs()
                    outFile.outputStream().use { output -> zis.copyTo(output) }
                }
                zis.closeEntry()
            }
        }
    }

    /** chmod 755 the given files. */
    private fun makeExecutable(vararg files: File) {
        files.forEach { file ->
            if (!file.exists()) return@forEach
            try {
                ProcessBuilder("/system/bin/chmod", "755", file.absolutePath)
                    .redirectErrorStream(true)
                    .start()
                    .waitFor()
            } catch (_: Exception) {}
            try {
                file.setExecutable(true, false)
            } catch (_: Exception) {}
        }
    }

    /**
     * Android 10+ exec() workaround:
     * Copy a binary into [nativeLibDir] with a `.so` suffix so the linker
     * treats it as a native library and SELinux allows execution.
     */
    private fun copyToNativeLibDir(source: File, destName: String) {
        if (!source.exists()) return
        val dest = File(nativeLibDir, destName)
        if (dest.exists() && dest.length() == source.length()) return
        try {
            source.inputStream().use { input ->
                dest.outputStream().use { output -> input.copyTo(output) }
            }
            dest.setExecutable(true, false)
            Timber.i("Copied ${source.name} -> nativeLibDir/$destName")
        } catch (e: Exception) {
            Timber.e(e, "Failed to copy ${source.name} to nativeLibDir")
        }
    }

    /**
     * On Android 10+, wrap a command so the binary is executed from
     * [nativeLibDir] (copied on-the-fly if needed).
     */
    private fun wrapForExec(cmd: List<String>): List<String> {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return cmd

        val binary = File(cmd[0])
        val targetName = "lib${binary.name.lowercase().replace("-", "_")}.so"
        val targetFile = File(nativeLibDir, targetName)

        if (!targetFile.exists() && binary.exists()) {
            copyToNativeLibDir(binary, targetName)
        }

        return if (targetFile.exists()) {
            listOf(targetFile.absolutePath) + cmd.drop(1)
        } else cmd
    }

    /** Read the last bytes of a dead process's combined stdout+stderr. */
    private fun readProcessError(process: Process?): String {
        return try {
            process?.inputStream?.bufferedReader()?.readText()?.take(500) ?: "Unknown"
        } catch (_: Exception) {
            "Unknown"
        }
    }
}
