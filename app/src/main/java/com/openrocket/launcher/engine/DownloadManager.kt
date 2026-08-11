package com.openrocket.launcher.engine

import android.content.Context
import com.openrocket.launcher.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.logging.HttpLoggingInterceptor
import timber.log.Timber
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.util.concurrent.TimeUnit

sealed class DownloadState {
    object Idle : DownloadState()
    data class Downloading(val progress: Float, val bytesDownloaded: Long, val totalBytes: Long) : DownloadState()
    object Completed : DownloadState()
    data class Failed(val error: String) : DownloadState()
    object Cancelled : DownloadState()
}

class DownloadManager private constructor(context: Context) {

    private val appContext = context.applicationContext
    private val client: OkHttpClient

    private val _downloadStates = mutableMapOf<String, MutableStateFlow<DownloadState>>()
    private val activeDownloads = mutableMapOf<String, Boolean>()

    init {
        val logging = HttpLoggingInterceptor().apply {
            level = if (BuildConfig.DEBUG) {
                HttpLoggingInterceptor.Level.BASIC
            } else {
                HttpLoggingInterceptor.Level.NONE
            }
        }

        client = OkHttpClient.Builder()
            .addInterceptor(logging)
            .connectTimeout(60, TimeUnit.SECONDS)
            .readTimeout(600, TimeUnit.SECONDS)   // 10 min for large JDK download
            .writeTimeout(60, TimeUnit.SECONDS)
            .build()
    }

    fun getDownloadState(key: String): StateFlow<DownloadState> {
        return _downloadStates.getOrPut(key) {
            MutableStateFlow(DownloadState.Idle)
        }.asStateFlow()
    }

    suspend fun download(
        url: String,
        destinationFile: File,
        key: String = url
    ): Result<File> = withContext(Dispatchers.IO) {
        val stateFlow = _downloadStates.getOrPut(key) { MutableStateFlow(DownloadState.Idle) }

        if (activeDownloads[key] == true) {
            Timber.w("Download already in progress for key: $key")
            return@withContext Result.failure(IOException("Download already in progress"))
        }

        activeDownloads[key] = true
        stateFlow.value = DownloadState.Downloading(0f, 0L, 0L)

        try {
            val tempFile = File(destinationFile.absolutePath + ".tmp")
            val resumePosition = if (tempFile.exists()) tempFile.length() else 0L

            val requestBuilder = Request.Builder()
                .url(url)
                .header("User-Agent", "OpenRocket-Launcher/1.0")

            if (resumePosition > 0) {
                requestBuilder.header("Range", "bytes=$resumePosition-")
                Timber.d("Resuming download from byte $resumePosition")
            }

            val request = requestBuilder.build()
            val response = client.newCall(request).execute()

            if (!response.isSuccessful && response.code != 206) {
                throw IOException("Unexpected response code: ${response.code}")
            }

            val body = response.body ?: throw IOException("Empty response body")
            val totalBytes = body.contentLength().let {
                if (it == -1L) -1L else it + resumePosition
            }

            val append = response.code == 206
            val output = if (append && tempFile.exists()) {
                RandomAccessFile(tempFile, "rw").apply { seek(length()) }
            } else {
                tempFile.delete()
                RandomAccessFile(tempFile, "rw")
            }

            body.byteStream().use { input ->
                val buffer = ByteArray(8192)
                var bytesDownloaded = resumePosition
                var read: Int

                try {
                    try {
                while (input.read(buffer).also { read = it } != -1) {
                        if (activeDownloads[key] != true) {
                            stateFlow.value = DownloadState.Cancelled
                            return@withContext Result.failure(IOException("Download cancelled"))
                        }

                        output.write(buffer, 0, read)
                        bytesDownloaded += read

                        val progress = if (totalBytes > 0) {
                            bytesDownloaded.toFloat() / totalBytes.toFloat()
                        } else {
                            -1f
                        }
                        stateFlow.value = DownloadState.Downloading(progress, bytesDownloaded, totalBytes)
                }
            } finally {
                output.close()
                    }
                } finally {
                    output.close()
                }
            }

            if (!tempFile.renameTo(destinationFile)) {
                tempFile.copyTo(destinationFile, overwrite = true)
                tempFile.delete()
            }
            stateFlow.value = DownloadState.Completed
            Timber.i("Download completed: ${destinationFile.absolutePath}")
            Result.success(destinationFile)

        } catch (e: Exception) {
            Timber.e(e, "Download failed for $url")
            stateFlow.value = DownloadState.Failed(e.message ?: "Unknown error")
            Result.failure(e)
        } finally {
            activeDownloads[key] = false
        }
    }

    fun cancelDownload(key: String) {
        activeDownloads[key] = false
    }

    companion object {
        @Volatile
        private var instance: DownloadManager? = null

        fun getInstance(context: Context): DownloadManager {
            return instance ?: synchronized(this) {
                instance ?: DownloadManager(context).also { instance = it }
            }
        }
    }
}
