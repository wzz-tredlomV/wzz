package com.openrocket.launcher.engine

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream

class JavaAppManager(private val context: Context) {

    private val jdkManager = JdkManager.getInstance(context)
    private val downloadManager = DownloadManager.getInstance(context)

    private val _setupState = MutableStateFlow<JavaAppSetupState>(JavaAppSetupState.Idle)
    val setupState: StateFlow<JavaAppSetupState> = _setupState.asStateFlow()

    val jdkDir: File
        get() = File(context.filesDir, "jdk")

    val appsDir: File
        get() = File(context.filesDir, "apps")

    val workingDir: File
        get() = File(context.filesDir, "workspace")

    init {
        appsDir.mkdirs()
        workingDir.mkdirs()
    }

    suspend fun checkSetup(): Boolean = withContext(Dispatchers.IO) {
        _setupState.value = JavaAppSetupState.Checking

        val jdkReady = jdkManager.checkJdkInstallation()

        if (jdkReady) {
            _setupState.value = JavaAppSetupState.Ready("")
            return@withContext true
        }

        _setupState.value = JavaAppSetupState.Idle
        false
    }

    suspend fun setupJdk(jdkUrl: String = DEFAULT_JDK_URL): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            if (!jdkManager.checkJdkInstallation()) {
                _setupState.value = JavaAppSetupState.DownloadingJdk(0f)

                // Collect JDK extracting progress
                val stateJob = launch {
                    jdkManager.jdkState.collect { jdkState ->
                        when (jdkState) {
                            is JdkSetupState.Extracting -> {
                                _setupState.value = JavaAppSetupState.Extracting(jdkState.progress, jdkState.currentEntry)
                            }
                            else -> {}
                        }
                    }
                }

                val jdkResult = jdkManager.downloadAndInstallJdk(jdkUrl) { progress ->
                    _setupState.value = JavaAppSetupState.DownloadingJdk(progress)
                }

                stateJob.cancel()

                if (jdkResult.isFailure) {
                    val error = jdkResult.exceptionOrNull()?.message ?: "JDK download failed"
                    _setupState.value = JavaAppSetupState.Error(error)
                    return@withContext Result.failure(Exception(error))
                }
            }

            // Installation done - test JDK version
            _setupState.value = JavaAppSetupState.Testing
            val version = jdkManager.getJdkVersion()
            Timber.i("JDK version test result: $version")

            _setupState.value = JavaAppSetupState.Ready(version ?: "Unknown version")
            Result.success(Unit)
        } catch (e: Exception) {
            Timber.e(e, "JDK setup failed")
            _setupState.value = JavaAppSetupState.Error(e.message ?: "JDK setup failed")
            Result.failure(e)
        }
    }

    /**
     * Install JDK from a local file URI.
     */
    suspend fun installJdkFromLocalFile(uri: Uri): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            _setupState.value = JavaAppSetupState.Extracting(0f, "")

            // Collect JDK extracting progress
            val stateJob = launch {
                jdkManager.jdkState.collect { jdkState ->
                    when (jdkState) {
                        is JdkSetupState.Extracting -> {
                            _setupState.value = JavaAppSetupState.Extracting(jdkState.progress, jdkState.currentEntry)
                        }
                        else -> {}
                    }
                }
            }

            val result = jdkManager.installFromLocalFile(uri)
            stateJob.cancel()

            result.fold(
                onSuccess = {
                    // Test JDK version after local installation
                    _setupState.value = JavaAppSetupState.Testing
                    val version = jdkManager.getJdkVersion()
                    Timber.i("JDK version test result: $version")
                    _setupState.value = JavaAppSetupState.Ready(version ?: "Unknown version")
                },
                onFailure = { error ->
                    _setupState.value = JavaAppSetupState.Error(
                        error.message ?: "Local JDK installation failed"
                    )
                }
            )

            result
        } catch (e: Exception) {
            Timber.e(e, "Local JDK installation failed")
            _setupState.value = JavaAppSetupState.Error(e.message ?: "Local installation failed")
            Result.failure(e)
        }
    }

    suspend fun downloadApp(appName: String, jarUrl: String): Result<File> = withContext(Dispatchers.IO) {
        try {
            _setupState.value = JavaAppSetupState.DownloadingApp(0f, appName)

            val appDir = File(appsDir, appName.replace(Regex("[^a-zA-Z0-9_-]"), "_"))
            appDir.mkdirs()

            val jarFile = File(appDir, "$safeName.jar")

            val downloadResult = downloadManager.download(jarUrl, jarFile, "app_download_$appName")

            downloadResult.fold(
                onSuccess = { file ->
                    _setupState.value = JavaAppSetupState.Ready("")
                    Timber.i("App $appName downloaded: ${file.absolutePath}")
                    Result.success(file)
                },
                onFailure = { error ->
                    _setupState.value = JavaAppSetupState.Error(error.message ?: "Download failed")
                    Timber.e(error, "App $appName download failed")
                    Result.failure(error)
                }
            )
        } catch (e: Exception) {
            _setupState.value = JavaAppSetupState.Error(e.message ?: "Download failed")
            Timber.e(e, "App $appName download failed")
            Result.failure(e)
        }
    }

    /**
     * Import a JAR file from user-selected URI (SAF)
     */
    suspend fun importJar(uri: Uri, appName: String? = null): Result<InstalledApp> = withContext(Dispatchers.IO) {
        try {
            val resolver = context.contentResolver
            val displayName = appName ?: getFileNameFromUri(uri) ?: "ImportedApp_${System.currentTimeMillis()}"
            val safeName = displayName.replace(Regex("[^a-zA-Z0-9_-]"), "_")

            val appDir = File(appsDir, safeName)
            appDir.mkdirs()

            val jarFile = File(appDir, "$safeName.jar")

            resolver.openInputStream(uri)?.use { input ->
                FileOutputStream(jarFile).use { output ->
                    input.copyTo(output)
                }
            } ?: return@withContext Result.failure(Exception("Cannot open input stream for URI"))

            // Also try to copy associated files if it's a document tree
            try {
                @Suppress("UNUSED_VARIABLE")
                val docUri = android.provider.DocumentsContract.buildDocumentUriUsingTree(
                    uri,
                    android.provider.DocumentsContract.getTreeDocumentId(uri)
                )
                // Best effort - copy main JAR is sufficient
            } catch (_: Exception) { }

            val app = InstalledApp(
                name = safeName,
                jarFile = jarFile,
                dir = appDir,
                isValid = jarFile.exists() && jarFile.length() > 0
            )

            Timber.i("Imported JAR: ${jarFile.absolutePath} (${jarFile.length()} bytes)")
            Result.success(app)
        } catch (e: Exception) {
            Timber.e(e, "Failed to import JAR from URI")
            Result.failure(e)
        }
    }

    /**
     * Import a JAR file from a local File path
     */
    fun importJarFile(sourceFile: File, appName: String? = null): InstalledApp? {
        return try {
            val displayName = appName ?: sourceFile.nameWithoutExtension
            val safeName = displayName.replace(Regex("[^a-zA-Z0-9_-]"), "_")

            val appDir = File(appsDir, safeName)
            appDir.mkdirs()

            val jarFile = File(appDir, "$safeName.jar")
            sourceFile.copyTo(jarFile, overwrite = true)

            InstalledApp(
                name = safeName,
                jarFile = jarFile,
                dir = appDir,
                isValid = jarFile.exists() && jarFile.length() > 0
            ).also {
                Timber.i("Imported JAR file: ${jarFile.absolutePath}")
            }
        } catch (e: Exception) {
            Timber.e(e, "Failed to import JAR file")
            null
        }
    }

    private fun getFileNameFromUri(uri: Uri): String? {
        var name: String? = null
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (nameIndex >= 0) {
                    name = cursor.getString(nameIndex)
                }
            }
        }
        return name
    }

    fun getInstalledApps(): List<InstalledApp> {
        return appsDir.listFiles()?.filter { it.isDirectory }?.map { dir ->
            val jarFile = dir.listFiles()?.find { it.extension == "jar" }
            InstalledApp(
                name = dir.name,
                jarFile = jarFile,
                dir = dir,
                isValid = jarFile != null && jarFile.exists() && jarFile.length() > 0
            )
        }?.sortedBy { it.name } ?: emptyList()
    }

    fun getAppJar(appName: String): File? {
        val appDir = File(appsDir, appName)
        return appDir.listFiles()?.find { it.extension == "jar" }
    }

    fun deleteApp(appName: String): Boolean {
        val appDir = File(appsDir, appName)
        return appDir.deleteRecursively().also {
            if (it) Timber.i("Deleted app: $appName")
        }
    }

    fun getJdkVersion(): String? = jdkManager.getJdkVersion()

    val javaBinary: File
        get() = jdkManager.javaBinary

    val javaHome: String
        get() = jdkManager.javaHome

    companion object {
        // OpenJDK 17 for Android ARM64 (prebuilt by itsaky)
        const val DEFAULT_JDK_URL =
            "https://github.com/itsaky/openjdk-17-android/releases/download/01-01-2022/jdk17-arm64.tar.xz"

        // Alternative: Use a direct OpenJDK tarball
        const val ALT_JDK_URL =
            "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.9%2B9/OpenJDK17U-jdk_aarch64_linux_hotspot_17.0.9_9.tar.gz"
    }
}

data class InstalledApp(
    val name: String,
    val jarFile: File?,
    val dir: File,
    val isValid: Boolean
)
