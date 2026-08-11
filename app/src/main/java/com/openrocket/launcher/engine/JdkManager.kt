package com.openrocket.launcher.engine

import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.xz.XZCompressorInputStream
import timber.log.Timber
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.IOException

sealed class JdkSetupState {
    object NotInstalled : JdkSetupState()
    object Checking : JdkSetupState()
    data class Downloading(val progress: Float) : JdkSetupState()
    data class Extracting(val progress: Float, val currentEntry: String = "") : JdkSetupState()
    object Ready : JdkSetupState()
    data class Error(val message: String) : JdkSetupState()
}

class JdkManager private constructor(private val context: Context) {

    companion object {
        @Volatile
        private var instance: JdkManager? = null

        fun getInstance(context: Context): JdkManager {
            return instance ?: synchronized(this) {
                instance ?: JdkManager(context.applicationContext).also { instance = it }
            }
        }

        const val DEFAULT_JDK_URL =
            "https://github.com/itsaky/openjdk-17-android/releases/download/01-01-2022/jdk17-arm64.tar.xz"

        const val ALT_JDK_URL =
            "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.9%2B9/OpenJDK17U-jdk_aarch64_linux_hotspot_17.0.9_9.tar.gz"
    }

    private val downloadManager = DownloadManager.getInstance(context)

    private val _jdkState = kotlinx.coroutines.flow.MutableStateFlow<JdkSetupState>(JdkSetupState.NotInstalled)
    val jdkState: StateFlow<JdkSetupState> = _jdkState.asStateFlow()

    private val jdkDir: File
        get() = File(context.filesDir, "jdk")

    /**
     * Find the actual java binary, handling tar archives that have a root directory.
     * Checks jdkDir/bin/java first, then searches one level deep.
     */
    val javaBinary: File
        get() = findJavaBinary()

    val javaHome: String
        get() = javaBinary.parentFile?.parentFile?.absolutePath ?: jdkDir.absolutePath

    /**
     * On Android 10+, the native library directory is the ONLY place where
     * apps can execute binaries without root. This directory has SELinux
     * context apk_data_file which allows exec.
     *
     * We copy the java binary (renamed as libjava.so) and libjli.so here
     * so they can be executed without triggering exec restrictions.
     */
    private val nativeLibDir: File by lazy {
        File(context.applicationInfo.nativeLibraryDir)
    }

    /**
     * Path to the java binary in nativeLibraryDir (for Android 10+ exec).
     * Returns null if not yet set up.
     */
    val nativeJavaBinary: File
        get() = File(nativeLibDir, "libjava.so")

    /**
     * Check if native library dir has our JDK binaries.
     */
    fun hasNativeBinaries(): Boolean {
        return nativeJavaBinary.exists() && nativeJavaBinary.length() > 0
    }

    /**
     * Copy critical JDK binaries to nativeLibraryDir for Android 10+ execution.
     * This is the KEY workaround for Android 10+ exec restrictions.
     */
    fun setupNativeBinaries(): Boolean {
        return try {
            val isAndroid10Plus = android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q
            if (!isAndroid10Plus) {
                Timber.d("Android 9 or below, nativeLibraryDir workaround not needed")
                return true
            }

            if (!isJdkInstalled()) {
                Timber.w("JDK not installed, cannot setup native binaries")
                return false
            }

            Timber.i("Setting up native binaries in ${nativeLibDir.absolutePath}")

            // Copy java binary as libjava.so
            val sourceJava = javaBinary
            if (sourceJava.exists()) {
                sourceJava.copyTo(nativeJavaBinary, overwrite = true)
                Timber.i("Copied java -> libjava.so (${nativeJavaBinary.length()} bytes)")
            }

            // Copy libjli.so (critical for JLI_Launch)
            val sourceJli = File(jdkDir, "lib/jli/libjli.so")
            val destJli = File(nativeLibDir, "libjli.so")
            if (sourceJli.exists()) {
                sourceJli.copyTo(destJli, overwrite = true)
                Timber.i("Copied libjli.so -> nativeLibDir")
            }

            // Copy other critical .so files
            val criticalLibs = listOf(
                "lib/server/libjvm.so" to "libjvm.so",
                "lib/libverify.so" to "libverify.so",
                "lib/libjava.so" to "libjavajdk.so",
                "lib/libnet.so" to "libnet.so",
                "lib/libnio.so" to "libnio.so",
                "lib/libzip.so" to "libzip.so",
                "lib/libjimage.so" to "libjimage.so"
            )

            for ((srcPath, destName) in criticalLibs) {
                val src = File(jdkDir, srcPath)
                val dst = File(nativeLibDir, destName)
                if (src.exists()) {
                    src.copyTo(dst, overwrite = true)
                    Timber.i("Copied $srcPath -> $destName")
                } else {
                    Timber.d("Source not found: $srcPath")
                }
            }

            Timber.i("Native binary setup complete")
            hasNativeBinaries()
        } catch (e: Exception) {
            Timber.e(e, "Failed to setup native binaries")
            false
        }
    }

    private fun findJavaBinary(): File {
        val direct = File(jdkDir, "bin/java")
        if (direct.exists()) return direct

        // Some tarballs have a single root directory (e.g., jdk-17.0.1/)
        jdkDir.listFiles()?.filter { it.isDirectory }?.forEach { subDir ->
            val nested = File(subDir, "bin/java")
            if (nested.exists()) return nested
        }
        return direct
    }

    suspend fun checkJdkInstallation(): Boolean = withContext(Dispatchers.IO) {
        _jdkState.value = JdkSetupState.Checking

        if (isJdkInstalled()) {
            Timber.i("JDK already installed at ${jdkDir.absolutePath}")
            _jdkState.value = JdkSetupState.Ready
            return@withContext true
        }

        Timber.i("JDK not found at ${jdkDir.absolutePath}")
        _jdkState.value = JdkSetupState.NotInstalled
        false
    }

    suspend fun downloadAndInstallJdk(
        jdkUrl: String = DEFAULT_JDK_URL,
        onProgress: (Float) -> Unit = {}
    ): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            _jdkState.value = JdkSetupState.Downloading(0f)

            val jdkArchiveFile = File(context.cacheDir, "openjdk17_arm64.tar.xz")

            if (jdkArchiveFile.exists()) {
                jdkArchiveFile.delete()
            }

            val downloadStateFlow = downloadManager.getDownloadState("jdk_download")
            var downloadSuccess = false
            var downloadError: String? = null

            val downloadJob = launch {
                val result = downloadManager.download(jdkUrl, jdkArchiveFile, "jdk_download")
                downloadSuccess = result.isSuccess
                downloadError = result.exceptionOrNull()?.message
            }

            while (isActive && downloadJob.isActive) {
                val state = downloadStateFlow.value
                when (state) {
                    is DownloadState.Downloading -> {
                        val progress = if (state.progress >= 0f) state.progress else 0f
                        _jdkState.value = JdkSetupState.Downloading(progress)
                        onProgress(progress)
                    }
                    is DownloadState.Completed -> {
                        _jdkState.value = JdkSetupState.Downloading(1f)
                        onProgress(1f)
                    }
                    is DownloadState.Failed -> {
                        _jdkState.value = JdkSetupState.Error(state.error)
                        onProgress(0f)
                    }
                    else -> { }
                }
                delay(500)
            }

            downloadJob.join()

            if (!downloadSuccess) {
                val error = downloadError ?: "Download failed"
                _jdkState.value = JdkSetupState.Error(error)
                return@withContext Result.failure(IOException(error))
            }
            if (!jdkArchiveFile.exists() || jdkArchiveFile.length() == 0L) {
                return@withContext Result.failure(IOException("Download failed: file not found or empty"))
            }

            Timber.i("Download complete: ${jdkArchiveFile.length()} bytes")

            if (jdkDir.exists()) {
                jdkDir.deleteRecursively()
            }

            _jdkState.value = JdkSetupState.Extracting(0f, "")
            val extractResult = extractArchive(
                archiveFile = jdkArchiveFile,
                destinationDir = jdkDir,
                onProgress = { progress, entry ->
                    _jdkState.value = JdkSetupState.Extracting(progress, entry)
                }
            )
            jdkArchiveFile.delete()

            if (extractResult.isFailure) {
                val error = extractResult.exceptionOrNull()?.message ?: "Extraction failed"
                _jdkState.value = JdkSetupState.Error(error)
                return@withContext Result.failure(IOException(error))
            }

            Timber.i("Extraction complete. Checking installation...")
            makeBinariesExecutable()

            if (isJdkInstalled()) {
                _jdkState.value = JdkSetupState.Ready
                Timber.i("JDK installation completed successfully")
                Result.success(Unit)
            } else {
                val files = jdkDir.listFiles()?.map { it.name } ?: emptyList()
                val binDir = File(jdkDir, "bin")
                val binFiles = if (binDir.exists()) binDir.listFiles()?.map { it.name } ?: emptyList() else emptyList()
                val error = "JDK extraction failed - java binary not found. " +
                        "Root files: $files, Bin exists: ${binDir.exists()}, Bin files: $binFiles"
                Timber.e(error)
                _jdkState.value = JdkSetupState.Error("JDK extraction failed - java binary not found")
                Result.failure(IOException(error))
            }
        } catch (e: OutOfMemoryError) {
            Timber.e(e, "OOM during JDK installation")
            _jdkState.value = JdkSetupState.Error("Out of memory during extraction")
            Result.failure(e)
        } catch (e: Exception) {
            Timber.e(e, "JDK installation failed")
            _jdkState.value = JdkSetupState.Error(e.message ?: "Installation failed")
            Result.failure(e)
        }
    }

    private fun isJdkInstalled(): Boolean {
        val binary = javaBinary
        val exists = binary.exists() && binary.length() > 0
        if (exists) {
            Timber.d("JDK found at ${binary.absolutePath}")
        } else {
            Timber.d("JDK not found. Searched: ${binary.absolutePath}")
            // List jdkDir contents for debugging
            jdkDir.listFiles()?.let { files ->
                Timber.d("jdkDir contents: ${files.map { it.name }}")
            }
        }
        return exists
    }

    /**
     * Extract tar.xz archive. Tries multiple strategies:
     * 1. Apache Commons Compress (most reliable when dependencies are present)
     * 2. System tar command (fallback)
     * 3. unxz + tar (last resort)
     */
    private fun extractArchive(
        archiveFile: File,
        destinationDir: File,
        onProgress: (Float, String) -> Unit = { _, _ -> }
    ): Result<Unit> {
        // Strategy 1: Apache Commons Compress with Tukaani XZ
        val manualResult = extractWithCommonsCompress(archiveFile, destinationDir, onProgress)
        if (manualResult.isSuccess) {
            Timber.i("Extraction succeeded with Apache Commons Compress")
            return manualResult
        }
        Timber.w("Commons Compress failed: ${manualResult.exceptionOrNull()?.message}")

        onProgress(0f, "Trying system tar...")
        // Strategy 2: System tar command
        val tarPaths = listOf("/system/bin/tar", "tar", "/bin/tar")
        for (tarPath in tarPaths) {
            val result = trySystemTar(tarPath, archiveFile, destinationDir)
            if (result.isSuccess) {
                Timber.i("Extraction succeeded with system tar ($tarPath)")
                onProgress(1f, "Done")
                return result
            }
            Timber.d("tar $tarPath failed: ${result.exceptionOrNull()?.message}")
        }

        onProgress(0f, "Trying unxz + tar...")
        // Strategy 3: unxz + tar
        val unxzResult = tryUnxzThenTar(archiveFile, destinationDir)
        if (unxzResult.isSuccess) {
            Timber.i("Extraction succeeded with unxz+tar")
            onProgress(1f, "Done")
            return unxzResult
        }
        Timber.w("unxz+tar failed: ${unxzResult.exceptionOrNull()?.message}")

        return Result.failure(IOException(
            "All extraction methods failed. " +
            "Commons Compress: ${manualResult.exceptionOrNull()?.message}. " +
            "unxz+tar: ${unxzResult.exceptionOrNull()?.message}"
        ))
    }

    private fun extractWithCommonsCompress(
        archiveFile: File,
        destinationDir: File,
        onProgress: (Float, String) -> Unit = { _, _ -> }
    ): Result<Unit> {
        var tis: TarArchiveInputStream? = null
        return try {
            destinationDir.mkdirs()
            val archiveLength = archiveFile.length()
            Timber.i("Extracting ${archiveLength} bytes with Commons Compress")

            // First pass: calculate total uncompressed bytes for accurate progress
            Timber.d("First pass: calculating total bytes...")
            var totalEntries = 0
            var totalBytes = 0L
            BufferedInputStream(FileInputStream(archiveFile), 65536).use { countInput ->
                val countXz = XZCompressorInputStream(countInput, false)
                val countTis = TarArchiveInputStream(countXz)
                var ce = countTis.nextEntry
                while (ce != null) {
                    totalEntries++
                    if (!ce.isDirectory && ce.size > 0) {
                        totalBytes += ce.size
                    }
                    ce = try { countTis.nextEntry } catch (_: Exception) { null }
                }
            }
            Timber.i("Total entries: $totalEntries, total bytes: $totalBytes")

            // Second pass: actual extraction with byte-based progress
            val fileInput = BufferedInputStream(FileInputStream(archiveFile), 65536)
            val xzInput = XZCompressorInputStream(fileInput, false)
            tis = TarArchiveInputStream(xzInput)

            val buffer = ByteArray(8192)
            var entry = tis.nextEntry
            var count = 0
            var fileCount = 0
            var extractedBytes = 0L

            while (entry != null) {
                try {
                    var name = entry.name.replace("\\", "/")
                    if (name.startsWith("./")) name = name.substring(2)
                    if (name == "." || name.isEmpty()) {
                        entry = tis.nextEntry
                        continue
                    }
                    if (name.startsWith("/") || name.contains("../")) {
                        entry = tis.nextEntry
                        continue
                    }

                    val outFile = File(destinationDir, name)
                    if (entry.isDirectory) {
                        outFile.mkdirs()
                        count++
                    } else {
                        outFile.parentFile?.mkdirs()
                        outFile.outputStream().use { output ->
                            var read: Int
                            while (tis.read(buffer).also { read = it } != -1) {
                                output.write(buffer, 0, read)
                                extractedBytes += read
                            }
                        }
                        count++
                        fileCount++
                    }

                    // Report progress based on bytes extracted
                    if (totalBytes > 0) {
                        val progress = (extractedBytes.toFloat() / totalBytes.toFloat()).coerceIn(0f, 1f)
                        onProgress(progress, name)
                    }
                } catch (e: Exception) {
                    Timber.e(e, "Failed entry: ${entry?.name}")
                }
                entry = try { tis.nextEntry } catch (_: Exception) { null }
            }

            onProgress(1f, "Done")
            Timber.i("Extraction complete: $count entries ($fileCount files)")
            if (count > 0 && fileCount > 0) {
                Result.success(Unit)
            } else {
                Result.failure(IOException("No files extracted (entries=$count, files=$fileCount)"))
            }
        } catch (e: NoClassDefFoundError) {
            Timber.e(e, "XZ library not found - add org.tukaani:xz dependency")
            Result.failure(IOException("XZ decompression library missing"))
        } catch (e: Throwable) {
            Timber.e(e, "Manual extraction failed: ${e.javaClass.name}")
            Result.failure(IOException("Extraction failed: ${e.message}"))
        } finally {
            try { tis?.close() } catch (_: Exception) {}
        }
    }

    private fun trySystemTar(tarPath: String, archiveFile: File, destinationDir: File): Result<Unit> {
        return try {
            destinationDir.mkdirs()

            val pb = ProcessBuilder(tarPath, "-xf", archiveFile.absolutePath, "-C", destinationDir.absolutePath)
                .redirectErrorStream(true)
            val process = pb.start()
            val output = process.inputStream.bufferedReader().readText()
            val exitCode = process.waitFor()

            Timber.d("tar exit=$exitCode output=$output")

            if (exitCode == 0 && destinationDir.listFiles()?.isNotEmpty() == true) {
                Result.success(Unit)
            } else {
                Result.failure(IOException("tar exit=$exitCode, files=${destinationDir.listFiles()?.size}, output=$output"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private fun tryUnxzThenTar(archiveFile: File, destinationDir: File): Result<Unit> {
        val tarFile = File(context.cacheDir, "openjdk17_arm64.tar")
        return try {
            if (tarFile.exists()) tarFile.delete()

            // unxz
            val unxzPaths = listOf("/system/bin/unxz", "unxz")
            var unxzOk = false
            for (path in unxzPaths) {
                try {
                    val pb = ProcessBuilder(path, "-k", "-c", archiveFile.absolutePath)
                        .redirectOutput(tarFile)
                        .redirectErrorStream(true)
                    val p = pb.start()
                    p.waitFor()
                    if (tarFile.exists() && tarFile.length() > 0) {
                        unxzOk = true
                        break
                    }
                } catch (_: Exception) {}
            }

            if (!unxzOk) {
                return Result.failure(IOException("unxz failed"))
            }

            // tar
            val tarPaths = listOf("/system/bin/tar", "tar")
            for (path in tarPaths) {
                try {
                    val pb = ProcessBuilder(path, "-xf", tarFile.absolutePath, "-C", destinationDir.absolutePath)
                        .redirectErrorStream(true)
                    val p = pb.start()
                    val out = p.inputStream.bufferedReader().readText()
                    val code = p.waitFor()
                    tarFile.delete()
                    if (code == 0 && destinationDir.listFiles()?.isNotEmpty() == true) {
                        return Result.success(Unit)
                    }
                    Timber.d("tar from unxz: $out")
                } catch (_: Exception) {}
            }
            tarFile.delete()
            Result.failure(IOException("tar after unxz failed"))
        } catch (e: Exception) {
            tarFile.delete()
            Result.failure(e)
        }
    }

    private fun makeBinariesExecutable() {
        try {
            val isAndroid10Plus = android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q
            if (isAndroid10Plus) {
                Timber.w("Android 10+ detected - app_data_file execute restrictions apply. " +
                        "Trying best-effort permission fixes.")
            }

            // Find the actual bin directory (handles tar archives with root directory)
            val binDir = findBinDir()
            if (binDir != null && binDir.exists() && binDir.isDirectory) {
                Timber.i("Making binaries executable in ${binDir.absolutePath}")
                binDir.listFiles()?.forEach { file ->
                    if (!file.name.endsWith(".dll") && !file.name.endsWith(".exe")) {
                        makeFileExecutable(file)
                    }
                }
            } else {
                Timber.w("bin directory not found in $jdkDir")
            }

            // Find and chmod jspawnhelper
            val jspawnhelper = findJspawnhelper()
            if (jspawnhelper != null && jspawnhelper.exists()) {
                Timber.i("Making jspawnhelper executable: ${jspawnhelper.absolutePath}")
                makeFileExecutable(jspawnhelper)
            }

            // Chmod the java binary itself
            val binary = javaBinary
            if (binary.exists()) {
                Timber.i("Making java binary executable: ${binary.absolutePath}")
                makeFileExecutable(binary)

                // On Android 10+, also try to copy java binary to /data/local/tmp as fallback
                if (isAndroid10Plus) {
                    tryCopyToTmp(binary)
                }
            }

            // Also make .so files in lib/ executable (needed for some JDK builds)
            val libDir = findLibDir()
            if (libDir != null && libDir.exists()) {
                libDir.listFiles()?.forEach { file ->
                    if (file.name.endsWith(".so")) {
                        makeFileExecutable(file)
                    }
                }
                // Also check lib/server/
                val libServerDir = File(libDir, "server")
                if (libServerDir.exists()) {
                    libServerDir.listFiles()?.forEach { file ->
                        if (file.name.endsWith(".so")) {
                            makeFileExecutable(file)
                        }
                    }
                }
            }

            // On Android 10+, try to fix SELinux context for all .so files
            // Note: This requires root on most devices, so we skip it for non-root
            // to avoid hanging the installation process
            if (isAndroid10Plus) {
                Timber.d("Skipping SELinux fix on non-root device")
            }
        } catch (e: Exception) {
            Timber.e(e, "chmod failed")
        }
    }

    /**
     * Make a single file executable using multiple methods.
     */
    private fun makeFileExecutable(file: File) {
        if (!file.exists()) return

        try {
            // Method 1: System chmod (most reliable)
            val process = ProcessBuilder("/system/bin/chmod", "755", file.absolutePath)
                .redirectErrorStream(true)
                .start()
            process.waitFor()
        } catch (e: Exception) {
            Timber.w("chmod failed for ${file.name}: ${e.message}")
        }

        try {
            // Method 2: Java setExecutable
            if (!file.canExecute()) {
                file.setExecutable(true, false)
            }
        } catch (e: Exception) {
            Timber.w("setExecutable failed for ${file.name}: ${e.message}")
        }
    }

    /**
     * On Android 10+, try to copy critical binaries to /data/local/tmp where SELinux
     * policies may be more permissive for execution.
     */
    private fun tryCopyToTmp(sourceFile: File) {
        try {
            val tmpDir = File("/data/local/tmp")
            if (!tmpDir.exists() || !tmpDir.canWrite()) {
                Timber.d("Cannot write to /data/local/tmp")
                return
            }

            val tmpFile = File(tmpDir, "java_launcher_${sourceFile.name}")
            sourceFile.copyTo(tmpFile, overwrite = true)

            // chmod 755
            val chmodProcess = ProcessBuilder("/system/bin/chmod", "755", tmpFile.absolutePath)
                .redirectErrorStream(true)
                .start()
            chmodProcess.waitFor()

            Timber.i("Copied ${sourceFile.name} to /data/local/tmp as fallback: ${tmpFile.absolutePath}")
        } catch (e: Exception) {
            Timber.w("Failed to copy to /data/local/tmp: ${e.message}")
        }
    }

    /**
     * Best-effort attempt to fix SELinux context for JDK files.
     * Uses chcon to set a more permissive context if available.
     */
    private fun tryFixSELinuxContext(dir: File) {
        try {
            // Try to use chcon to change context to something more permissive
            // This requires root on most devices, but we try anyway
            val chconProcess = ProcessBuilder(
                "/system/bin/chcon",
                "-R",
                "u:object_r:app_data_file:s0",
                dir.absolutePath
            ).redirectErrorStream(true).start()
            chconProcess.waitFor()
            Timber.d("chcon attempt completed")
        } catch (e: Exception) {
            Timber.d("chcon not available or failed: ${e.message}")
        }

        // Also try restorecon
        try {
            val restoreconProcess = ProcessBuilder(
                "/system/bin/restorecon",
                "-R",
                dir.absolutePath
            ).redirectErrorStream(true).start()
            restoreconProcess.waitFor()
            Timber.d("restorecon attempt completed")
        } catch (e: Exception) {
            Timber.d("restorecon not available or failed: ${e.message}")
        }
    }

    private fun findLibDir(): File? {
        val direct = File(jdkDir, "lib")
        if (direct.exists() && direct.isDirectory) return direct

        jdkDir.listFiles()?.filter { it.isDirectory }?.forEach { subDir ->
            val nested = File(subDir, "lib")
            if (nested.exists() && nested.isDirectory) return nested
        }
        return null
    }

    private fun findBinDir(): File? {
        val direct = File(jdkDir, "bin")
        if (direct.exists() && direct.isDirectory) return direct

        jdkDir.listFiles()?.filter { it.isDirectory }?.forEach { subDir ->
            val nested = File(subDir, "bin")
            if (nested.exists() && nested.isDirectory) return nested
        }
        return null
    }

    private fun findJspawnhelper(): File? {
        val direct = File(jdkDir, "lib/jspawnhelper")
        if (direct.exists()) return direct

        jdkDir.listFiles()?.filter { it.isDirectory }?.forEach { subDir ->
            val nested = File(subDir, "lib/jspawnhelper")
            if (nested.exists()) return nested
        }
        return null
    }

    /**
     * Install JDK from a local file URI (e.g., user-selected via file picker).
     * Supports .tar.xz and .tar.gz formats.
     */
    suspend fun installFromLocalFile(uri: Uri): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            _jdkState.value = JdkSetupState.Extracting(0f, "")

            val fileName = getFileNameFromUri(uri) ?: "jdk_archive.tar.xz"
            val localFile = File(context.cacheDir, fileName)

            // Copy from content URI to local file
            Timber.i("Copying JDK from $uri to ${localFile.absolutePath}")
            context.contentResolver.openInputStream(uri)?.use { input ->
                localFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            } ?: return@withContext Result.failure(IOException("Cannot open input stream for $uri"))

            Timber.i("Copied ${localFile.length()} bytes")

            // Clean old JDK dir
            if (jdkDir.exists()) {
                jdkDir.deleteRecursively()
            }

            // Extract
            val extractResult = extractArchive(
                archiveFile = localFile,
                destinationDir = jdkDir,
                onProgress = { progress, entry ->
                    _jdkState.value = JdkSetupState.Extracting(progress, entry)
                }
            )
            localFile.delete()

            if (extractResult.isFailure) {
                val error = extractResult.exceptionOrNull()?.message ?: "Extraction failed"
                _jdkState.value = JdkSetupState.Error(error)
                return@withContext Result.failure(IOException(error))
            }

            Timber.i("Extraction complete. Checking installation...")
            makeBinariesExecutable()

            if (isJdkInstalled()) {
                _jdkState.value = JdkSetupState.Ready
                Timber.i("JDK installation from local file completed successfully")
                Result.success(Unit)
            } else {
                val files = jdkDir.listFiles()?.map { it.name } ?: emptyList()
                val binDir = File(jdkDir, "bin")
                val binFiles = if (binDir.exists()) binDir.listFiles()?.map { it.name } ?: emptyList() else emptyList()
                val error = "JDK extraction failed - java binary not found. " +
                        "Root files: $files, Bin exists: ${binDir.exists()}, Bin files: $binFiles"
                Timber.e(error)
                _jdkState.value = JdkSetupState.Error("JDK extraction failed - java binary not found")
                Result.failure(IOException(error))
            }
        } catch (e: OutOfMemoryError) {
            Timber.e(e, "OOM during local JDK installation")
            _jdkState.value = JdkSetupState.Error("Out of memory during extraction")
            Result.failure(e)
        } catch (e: Exception) {
            Timber.e(e, "Local JDK installation failed")
            _jdkState.value = JdkSetupState.Error(e.message ?: "Local installation failed")
            Result.failure(e)
        }
    }

    private fun getFileNameFromUri(uri: Uri): String? {
        return when (uri.scheme) {
            "content" -> {
                context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                    if (cursor.moveToFirst()) {
                        val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                        if (nameIndex >= 0) cursor.getString(nameIndex) else null
                    } else null
                }
            }
            "file" -> uri.lastPathSegment
            else -> null
        }
    }

    fun getJdkVersion(): String? {
        // Strategy 1: Read release file (most reliable, no process execution needed)
        val versionFromRelease = readVersionFromReleaseFile()
        if (versionFromRelease != null) {
            Timber.i("JDK version from release file: $versionFromRelease")
            return versionFromRelease
        }

        // Strategy 2: Execute java -version (with Android 10+ fallback strategies)
        return try {
            if (!isJdkInstalled()) return null

            val binary = javaBinary
            val libDir = File(jdkDir, "lib")
            val libServerDir = File(libDir, "server")
            val ldPaths = mutableListOf<String>()
            if (libDir.exists()) ldPaths.add(libDir.absolutePath)
            if (libServerDir.exists()) ldPaths.add(libServerDir.absolutePath)

            // Try multiple execution strategies for java -version
            val strategies = listOf(
                // Direct execution
                listOf(binary.absolutePath, "-version"),
                // Via shell
                listOf("/system/bin/sh", "-c", "${binary.absolutePath} -version"),
                // Via linker64 (Android 10+)
                listOf("/system/bin/linker64", binary.absolutePath, "-version")
            )

            for (cmd in strategies) {
                try {
                    val processBuilder = ProcessBuilder(cmd)
                        .redirectErrorStream(true)

                    if (ldPaths.isNotEmpty()) {
                        processBuilder.environment()["LD_LIBRARY_PATH"] = ldPaths.joinToString(":")
                    }

                    val process = processBuilder.start()
                    val reader = process.inputStream.bufferedReader()
                    val output = StringBuilder()
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        output.append(line).append("\n")
                    }
                    process.waitFor()

                    val outputStr = output.toString().trim()
                    if (outputStr.isNotEmpty()) {
                        // Parse version from output like: openjdk version "17.0.1" 2021-10-19
                        val versionRegex = Regex("""version\s+"([^"]+)""")
                        val match = versionRegex.find(outputStr)
                        val version = match?.groupValues?.get(1) ?: outputStr.lineSequence().firstOrNull()
                        if (version != null) {
                            Timber.i("JDK version from java binary (strategy: ${cmd.first()}): $version")
                            return version
                        }
                    }
                } catch (e: Exception) {
                    Timber.w("Version check strategy ${cmd.first()} failed: ${e.message}")
                }
            }

            Timber.w("All version check strategies failed")
            null
        } catch (e: Exception) {
            Timber.e(e, "Failed to get JDK version")
            null
        }
    }

    private fun readVersionFromReleaseFile(): String? {
        return try {
            val releaseFile = File(jdkDir, "release")
            if (!releaseFile.exists()) {
                // Some tarballs have a single root directory, check one level deeper
                jdkDir.listFiles()?.filter { it.isDirectory }?.forEach { subDir ->
                    val nestedRelease = File(subDir, "release")
                    if (nestedRelease.exists()) {
                        return parseReleaseFile(nestedRelease)
                    }
                }
                return null
            }
            parseReleaseFile(releaseFile)
        } catch (e: Exception) {
            Timber.e(e, "Failed to read release file")
            null
        }
    }

    private fun parseReleaseFile(releaseFile: File): String? {
        return try {
            releaseFile.readText().lineSequence()
                .map { it.trim() }
                .find { it.startsWith("JAVA_VERSION=") }
                ?.substringAfter("JAVA_VERSION=")
                ?.trim('"', '\'')
        } catch (e: Exception) {
            Timber.e(e, "Failed to parse release file")
            null
        }
    }
}
