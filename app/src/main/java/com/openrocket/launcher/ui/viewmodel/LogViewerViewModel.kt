package com.openrocket.launcher.ui.viewmodel

import android.app.Application
import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.openrocket.launcher.engine.JarExecutor
import com.openrocket.launcher.engine.ProcessState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import timber.log.Timber
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class LogViewerViewModel(application: Application) : AndroidViewModel(application) {

    private val jarExecutor = JarExecutor.getInstance(application)

    val logs: StateFlow<String> = jarExecutor.logs
    val processState: StateFlow<ProcessState> = jarExecutor.processState

    fun clearLogs() {
        jarExecutor.clearLogs()
        Timber.d("Logs cleared by user")
    }

    fun shareLogs(context: Context) {
        viewModelScope.launch {
            try {
                val logsDir = File(context.cacheDir, "logs")
                logsDir.mkdirs()

                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
                val logFile = File(logsDir, "openrocket_logs_$timestamp.txt")
                logFile.writeText(jarExecutor.getLogs())

                val uri = FileProvider.getUriForFile(
                    context,
                    "${context.packageName}.fileprovider",
                    logFile
                )

                val intent = Intent(Intent.ACTION_SEND).apply {
                    type = "text/plain"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    putExtra(Intent.EXTRA_SUBJECT, "OpenRocket Logs $timestamp")
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }

                val chooser = Intent.createChooser(intent, "Share Logs")
                chooser.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(chooser)

                Timber.i("Shared logs file: ${logFile.name}")
            } catch (e: Exception) {
                Timber.e(e, "Failed to share logs")
            }
        }
    }
}
