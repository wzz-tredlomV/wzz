package com.openrocket.launcher.engine

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.io.File

class OpenRocketManager(private val context: Context) {

    private val downloadManager = DownloadManager.getInstance(context)
    private val javaAppManager = JavaAppManager(context)

    private val _downloadState = MutableStateFlow<DownloadState>(DownloadState.Idle)
    val downloadState: StateFlow<DownloadState> = _downloadState.asStateFlow()

    suspend fun downloadOpenRocket(url: String = DEFAULT_OPENROCKET_URL): Result<File> = withContext(Dispatchers.IO) {
        try {
            _downloadState.value = DownloadState.Downloading(0f)

            val jarFile = File(context.filesDir, "apps/openrocket/openrocket.jar")
            jarFile.parentFile?.mkdirs()

            val result = downloadManager.download(url, jarFile, "openrocket_download")

            result.fold(
                onSuccess = { file ->
                    _downloadState.value = DownloadState.Completed
                    Timber.i("OpenRocket downloaded to ${file.absolutePath}")
                    Result.success(file)
                },
                onFailure = { error ->
                    _downloadState.value = DownloadState.Failed(error.message ?: "Download failed")
                    Timber.e(error, "OpenRocket download failed")
                    Result.failure(error)
                }
            )
        } catch (e: Exception) {
            _downloadState.value = DownloadState.Failed(e.message ?: "Unknown error")
            Timber.e(e, "OpenRocket download failed")
            Result.failure(e)
        }
    }

    fun getOpenRocketJar(): File? {
        val jarFile = File(context.filesDir, "apps/openrocket/openrocket.jar")
        return if (jarFile.exists()) jarFile else null
    }

    companion object {
        const val DEFAULT_OPENROCKET_URL =
            "https://github.com/openrocket/openrocket/releases/download/release-23.09/OpenRocket-23.09.jar"
    }

    sealed class DownloadState {
        object Idle : DownloadState()
        data class Downloading(val progress: Float) : DownloadState()
        object Completed : DownloadState()
        data class Failed(val message: String) : DownloadState()
    }
}
