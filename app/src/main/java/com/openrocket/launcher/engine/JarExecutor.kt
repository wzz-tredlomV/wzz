package com.openrocket.launcher.engine

import android.content.Context
import android.os.Build
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.io.BufferedReader
import java.io.File
import java.io.IOException
import java.io.InputStreamReader

sealed class ProcessState {
    object Idle : ProcessState()
    object Starting : ProcessState()
    object Running : ProcessState()
    object Stopping : ProcessState()
    data class Exited(val exitCode: Int) : ProcessState()
    data class Error(val message: String) : ProcessState()
}

data class ProcessOutput(
    val stdout: String = "",
    val stderr: String = "",
    val timestamp: Long = System.currentTimeMillis()
)

private sealed class ExecutionStrategy {
    abstract fun buildCommand(
        javaBinary: File,
        jarFile: File,
        jvmArgs: List<String>,
        appArgs: List<String>
    ): List<String>

    data object Direct : ExecutionStrategy() {
        override fun buildCommand(
            javaBinary: File, jarFile: File, jvmArgs: List<String>, appArgs: List<String>
        ): List<String> = buildList {
            add(javaBinary.absolutePath)
            addAll(jvmArgs)
            add("-jar")
            add(jarFile.absolutePath)
            addAll(appArgs)
        }
    }

    data object ViaShell : ExecutionStrategy() {
        override fun buildCommand(
            javaBinary: File, jarFile: File, jvmArgs: List<String>, appArgs: List<String>
        ): List<String> {
            val cmd = buildString {
                append(javaBinary.absolutePath)
                jvmArgs.forEach { append(" \"$it\"") }
                append(" -jar \"${jarFile.absolutePath}\"")
                appArgs.forEach { append(" \"$it\"") }
            }
            return listOf("/system/bin/sh", "-c", cmd)
        }
    }

    data object ViaLinker : ExecutionStrategy() {
        override fun buildCommand(
            javaBinary: File, jarFile: File, jvmArgs: List<String>, appArgs: List<String>
        ): List<String> = buildList {
            add("/system/bin/linker64")
            add(javaBinary.absolutePath)
            addAll(jvmArgs)
            add("-jar")
            add(jarFile.absolutePath)
            addAll(appArgs)
        }
    }

    /**
     * Execute via nativeLibraryDir (Android 10+ primary strategy).
     * Copies java binary to nativeLibraryDir as libjava.so where SELinux
     * allows exec. Uses JLI_Launch via JNI to avoid exec() entirely.
     */
    data object ViaNativeLibDir : ExecutionStrategy() {
        override fun buildCommand(
            javaBinary: File, jarFile: File, jvmArgs: List<String>, appArgs: List<String>
        ): List<String> = emptyList() // Handled by JNI path
    }

    data object ViaJniJvm : ExecutionStrategy() {
        override fun buildCommand(
            javaBinary: File, jarFile: File, jvmArgs: List<String>, appArgs: List<String>
        ): List<String> = emptyList()
    }
}
class JarExecutor private constructor(context: Context) {

    companion object {
        @Volatile
        private var instance: JarExecutor? = null

        fun getInstance(context: Context): JarExecutor {
            return instance ?: synchronized(this) {
                instance ?: JarExecutor(context.applicationContext).also { instance = it }
            }
        }
    }

    private val context: Context = context.applicationContext
    private var currentProcess: Process? = null
    private var jniJvmLoader: JniJvmLoader? = null
    private val logBuffer = StringBuilder()
    private val maxLogSize = 500_000

    private val _processState = MutableStateFlow<ProcessState>(ProcessState.Idle)
    val processState: StateFlow<ProcessState> = _processState.asStateFlow()

    private val _logs = MutableStateFlow("")
    val logs: StateFlow<String> = _logs.asStateFlow()

    private val _lastOutput = MutableStateFlow(ProcessOutput())
    val lastOutput: StateFlow<ProcessOutput> = _lastOutput.asStateFlow()

    private fun detectStrategies(javaBinary: File): List<ExecutionStrategy> {
        val strategies = mutableListOf<ExecutionStrategy>()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // Android 10+: nativeLibraryDir is the ONLY reliable way to exec without root
            strategies.add(ExecutionStrategy.ViaNativeLibDir)
            strategies.add(ExecutionStrategy.ViaJniJvm)
        }

        // Fallback strategies (may work on some devices)
        strategies.add(ExecutionStrategy.Direct)
        if (File("/system/bin/sh").exists()) {
            strategies.add(ExecutionStrategy.ViaShell)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && File("/system/bin/linker64").exists()) {
            strategies.add(ExecutionStrategy.ViaLinker)
        }

        Timber.i("Detected ${strategies.size} execution strategies for Android ${Build.VERSION.SDK_INT}")
        return strategies
    }

    suspend fun startJar(
        javaBinary: File,
        jarFile: File,
        workingDir: File? = null,
        jvmArgs: List<String> = emptyList(),
        appArgs: List<String> = emptyList(),
        envVars: Map<String, String> = emptyMap()
    ): Result<Process> = withContext(Dispatchers.IO) {
        try {
            if (_processState.value is ProcessState.Running) {
                return@withContext Result.failure(IllegalStateException("Process already running"))
            }

            _processState.value = ProcessState.Starting
            clearLogs()
            ensureExecutable(javaBinary)

            val strategies = detectStrategies(javaBinary)
            var lastException: Exception? = null

            for (strategy in strategies) {
                try {
                    if (strategy is ExecutionStrategy.ViaNativeLibDir) {
                        Timber.i("Trying nativeLibraryDir strategy")
                        appendLog(">>> Trying execution strategy: nativeLibraryDir (Android 10+ primary)\n")

                        val jdkManager = JdkManager.getInstance(context)
                        if (!jdkManager.hasNativeBinaries()) {
                            Timber.i("Native binaries not found, setting up...")
                            appendLog(">>> Setting up native binaries in nativeLibraryDir...\n")
                            if (!jdkManager.setupNativeBinaries()) {
                                Timber.w("Failed to setup native binaries")
                                appendLog(">>> Failed to setup native binaries\n")
                                lastException = IllegalStateException("Failed to setup native binaries")
                                continue
                            }
                        }

                        val nativeJava = jdkManager.nativeJavaBinary
                        if (!nativeJava.exists()) {
                            Timber.e("Native java binary not found after setup")
                            lastException = IllegalStateException("Native java binary missing")
                            continue
                        }

                        val javaHome = javaBinary.parentFile?.parentFile?.absolutePath
                            ?: return@withContext Result.failure(IllegalStateException("Cannot determine JAVA_HOME"))

                        // Build command using nativeLibraryDir binary
                        val command = buildList {
                            add(nativeJava.absolutePath)
                            addAll(jvmArgs)
                            add("-jar")
                            add(jarFile.absolutePath)
                            addAll(appArgs)
                        }

                        Timber.i("Executing via nativeLibraryDir: ${command.joinToString(" ")}")
                        appendLog(">>> Command: ${command.joinToString(" ")}\n")

                        val processBuilder = ProcessBuilder(command)
                            .redirectErrorStream(true)

                        workingDir?.let { processBuilder.directory(it) }

                        val env = processBuilder.environment()
                        env.putAll(envVars)
                        env["JAVA_HOME"] = javaHome
                        env["LD_LIBRARY_PATH"] = "$javaHome/lib:$javaHome/lib/server:$javaHome/lib/jli"

                        val process = processBuilder.start()
                        currentProcess = process
                        _processState.value = ProcessState.Running

                        withContext(Dispatchers.IO) {
                            readProcessOutput(process)
                        }

                        Timber.i("nativeLibraryDir strategy succeeded!")
                        appendLog(">>> Execution strategy nativeLibraryDir succeeded!\n")
                        return@withContext Result.success(process)
                    }

                    if (strategy is ExecutionStrategy.ViaJniJvm) {
                        Timber.i("Trying JNI JLI_Launch strategy")
                        appendLog(">>> Trying execution strategy: JNI JLI_Launch\n")

                        val jniLoader = getJniJvmLoader()
                        val javaHome = javaBinary.parentFile?.parentFile?.absolutePath
                            ?: return@withContext Result.failure(IllegalStateException("Cannot determine JAVA_HOME"))

                        val mainClass = JarManifestParser.getMainClass(jarFile)
                        if (mainClass == null) {
                            Timber.e("Could not determine Main-Class from JAR manifest")
                            appendLog(">>> FAILED: Could not determine Main-Class from JAR manifest\n")
                            lastException = IllegalStateException("No Main-Class found in JAR manifest")
                            continue
                        }

                        Timber.i("Starting JAR via JNI JLI_Launch: mainClass=$mainClass, javaHome=$javaHome")

                        val exitCode = jniLoader.runJar(
                            javaHome = javaHome,
                            jarPath = jarFile.absolutePath,
                            mainClass = mainClass,
                            jvmOptions = jvmArgs.toTypedArray(),
                            appArgs = appArgs.toTypedArray()
                        )

                        _processState.value = if (exitCode == 0) {
                            ProcessState.Exited(exitCode)
                        } else {
                            ProcessState.Error("JLI_Launch exited with code $exitCode")
                        }
                        currentProcess = null

                        Timber.i("JLI_Launch strategy completed with exit code: $exitCode")
                        appendLog(">>> JLI_Launch execution completed with exit code: $exitCode\n")

                        return@withContext Result.success(ProcessBuilder("true").start().also { it.waitFor() })
                    }

                    val command = strategy.buildCommand(javaBinary, jarFile, jvmArgs, appArgs)
                    Timber.i("Trying strategy ${strategy.javaClass.simpleName}: ${command.joinToString(" ")}")
                    appendLog(">>> Trying execution strategy: ${strategy.javaClass.simpleName}\n")
                    appendLog(">>> Command: ${command.joinToString(" ")}\n")

                    val processBuilder = ProcessBuilder(command)
                        .redirectErrorStream(true)

                    workingDir?.let { processBuilder.directory(it) }

                    val env = processBuilder.environment()
                    env.putAll(envVars)

                    val javaHome = javaBinary.parentFile?.parentFile?.absolutePath
                    javaHome?.let { home ->
                        env["JAVA_HOME"] = home
                        val libDir = File(home, "lib")
                        val libServerDir = File(libDir, "server")
                        val ldPaths = mutableListOf<String>()
                        if (libDir.exists()) ldPaths.add(libDir.absolutePath)
                        if (libServerDir.exists()) ldPaths.add(libServerDir.absolutePath)

                        if (ldPaths.isNotEmpty()) {
                            val currentLdPath = env["LD_LIBRARY_PATH"] ?: ""
                            val newLdPath = if (currentLdPath.isNotEmpty()) {
                                "${ldPaths.joinToString(":")}:$currentLdPath"
                            } else {
                                ldPaths.joinToString(":")
                            }
                            env["LD_LIBRARY_PATH"] = newLdPath
                        }
                        Timber.d("Set JAVA_HOME=$home, LD_LIBRARY_PATH=${env["LD_LIBRARY_PATH"]}")
                    }

                    Timber.i("JAVA_HOME=${env["JAVA_HOME"]}")
                    Timber.i("LD_LIBRARY_PATH=${env["LD_LIBRARY_PATH"]}")
                    Timber.i("PATH=${env["PATH"]}")

                    val process = processBuilder.start()
                    currentProcess = process
                    _processState.value = ProcessState.Running

                    withContext(Dispatchers.IO) {
                        readProcessOutput(process)
                    }

                    Timber.i("Strategy ${strategy.javaClass.simpleName} succeeded!")
                    appendLog(">>> Execution strategy ${strategy.javaClass.simpleName} succeeded!\n")
                    return@withContext Result.success(process)

                } catch (e: Exception) {
                    lastException = e
                    val strategyName = strategy.javaClass.simpleName
                    Timber.w(e, "Strategy $strategyName failed: ${e.message}")
                    appendLog(">>> Strategy $strategyName failed: ${e.message}\n")
                }
            }

            val androidVersion = Build.VERSION.SDK_INT
            val isAndroid10Plus = androidVersion >= Build.VERSION_CODES.Q
            val errorMsg = buildString {
                append("Failed to start Java process. ")
                append("All ${strategies.size} execution strategies failed. ")
                append("Last error: ${lastException?.message ?: "Unknown"}. ")
                if (isAndroid10Plus) {
                    append("Android 10+ (API $androidVersion) restricts executing binaries from app directories. ")
                    append("Solutions: 1) Use a rooted device with SELinux permissive mode, ")
                    append("2) Use a device with Android 9 or below.")
                }
            }

            Timber.e(lastException, errorMsg)
            appendLog(">>> FAILED TO START: $errorMsg\n")
            _processState.value = ProcessState.Error(errorMsg)
            Result.failure(lastException ?: IOException(errorMsg))

        } catch (e: Exception) {
            Timber.e(e, "Failed to start JAR process: ${e.javaClass.name}: ${e.message}")
            appendLog(">>> FAILED TO START: ${e.javaClass.name}: ${e.message}\n")
            _processState.value = ProcessState.Error(e.message ?: "Failed to start process")
            Result.failure(e)
        }
    }

    private fun getJniJvmLoader(): JniJvmLoader {
        if (jniJvmLoader == null) {
            jniJvmLoader = JniJvmLoader()
        }
        return jniJvmLoader!!
    }

    private suspend fun readProcessOutput(process: Process) = withContext(Dispatchers.IO) {
        try {
            BufferedReader(InputStreamReader(process.inputStream)).use { reader ->
                var line: String?
                while (reader.readLine().also { line = it } != null) {
                    line?.let {
                        appendLog(it + "\n")
                        _lastOutput.value = ProcessOutput(stdout = it, timestamp = System.currentTimeMillis())
                        _lastOutput.value = ProcessOutput(stdout = it, timestamp = System.currentTimeMillis())
                        Timber.v(it)
                    }
                }
            }

            val exitCode = process.waitFor()
            Timber.i("Process exited with code: $exitCode")
            appendLog(">>> Process exited with code: $exitCode\n")

            _processState.value = ProcessState.Exited(exitCode)
            currentProcess = null
        } catch (e: Exception) {
            if (e.message?.contains("Stream closed") != true) {
                Timber.e(e, "Error reading process output")
            }
        }
    }

    suspend fun stopProcess(force: Boolean = false) = withContext(Dispatchers.IO) {
        val process = currentProcess ?: return@withContext

        _processState.value = ProcessState.Stopping
        appendLog(">>> Stopping process...\n")

        try {
            if (force) {
                process.destroyForcibly()
            } else {
                process.destroy()
            }

            if (!process.waitFor(5, java.util.concurrent.TimeUnit.SECONDS)) {
                process.destroyForcibly()
            }

            currentProcess = null
            Timber.i("Process stopped")
        } catch (e: Exception) {
            Timber.e(e, "Error stopping process")
        }
    }

    fun isRunning(): Boolean {
        return currentProcess?.isAlive == true
    }

    private fun ensureExecutable(file: File) {
        if (!file.exists()) {
            Timber.w("File does not exist: ${file.absolutePath}")
            return
        }

        Timber.i("Ensuring executable: ${file.absolutePath}")

        try {
            val process = ProcessBuilder("/system/bin/chmod", "755", file.absolutePath)
                .redirectErrorStream(true)
                .start()
            process.waitFor()
            Timber.i("chmod 755 applied to ${file.name}")
        } catch (e: Exception) {
            Timber.w("chmod failed: ${e.message}")
        }

        try {
            if (!file.canExecute()) {
                file.setExecutable(true, false)
                Timber.i("setExecutable applied to ${file.name}")
            }
        } catch (e: Exception) {
            Timber.w("setExecutable failed: ${e.message}")
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            try {
                val tmpDir = File("/data/local/tmp")
                if (tmpDir.exists() && tmpDir.canWrite()) {
                    val tmpFile = File(tmpDir, "java_launcher_${file.name}")
                    file.copyTo(tmpFile, overwrite = true)
                    val chmodProcess = ProcessBuilder("/system/bin/chmod", "755", tmpFile.absolutePath)
                        .redirectErrorStream(true)
                        .start()
                    chmodProcess.waitFor()
                    Timber.i("Copied ${file.name} to /data/local/tmp as fallback")
                }
            } catch (e: Exception) {
                Timber.w("/data/local/tmp fallback failed: ${e.message}")
            }
        }

        Timber.i("Executable status: ${file.canExecute()}")
    }

    private fun appendLog(line: String) {
        synchronized(logBuffer) {
            logBuffer.append(line)
            if (logBuffer.length > maxLogSize) {
                logBuffer.delete(0, logBuffer.length - maxLogSize)
            }
            _logs.value = logBuffer.toString()
        }
    }

    fun clearLogs() {
        synchronized(logBuffer) {
            logBuffer.clear()
            _logs.value = ""
        }
    }

    fun getLogs(): String {
        return synchronized(logBuffer) {
            logBuffer.toString()
        }
    }
}
