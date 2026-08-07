package com.openrocket.launcher.ui.viewmodel

import android.app.Application
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.openrocket.launcher.engine.InstalledApp
import com.openrocket.launcher.engine.JavaAppManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import timber.log.Timber
import java.io.File

class FileBrowserViewModel(application: Application) : AndroidViewModel(application) {

    private val javaAppManager = JavaAppManager(application)

    private val _apps = MutableStateFlow<List<InstalledApp>>(emptyList())
    val apps: StateFlow<List<InstalledApp>> = _apps.asStateFlow()

    private val _workspaceFiles = MutableStateFlow<List<File>>(emptyList())
    val workspaceFiles: StateFlow<List<File>> = _workspaceFiles.asStateFlow()

    private val _importStatus = MutableStateFlow<String?>(null)
    val importStatus: StateFlow<String?> = _importStatus.asStateFlow()

    init {
        refresh()
    }

    fun refresh() {
        _apps.value = javaAppManager.getInstalledApps()
        _workspaceFiles.value = javaAppManager.workingDir.listFiles()?.toList()?.sortedByDescending { it.lastModified() } ?: emptyList()
    }

    fun importJar(uri: Uri) {
        viewModelScope.launch {
            _importStatus.value = "Importing..."
            val result = javaAppManager.importJar(uri)
            result.fold(
                onSuccess = { app ->
                    _importStatus.value = "Imported: ${app.name}"
                    refresh()
                },
                onFailure = { error ->
                    _importStatus.value = "Import failed: ${error.message}"
                    Timber.e(error, "JAR import failed")
                }
            )
        }
    }

    fun deleteApp(appName: String) {
        viewModelScope.launch {
            if (javaAppManager.deleteApp(appName)) {
                Timber.i("Deleted app: $appName")
                refresh()
            } else {
                Timber.w("Failed to delete app: $appName")
            }
        }
    }

    fun createWorkspaceFile(fileName: String) {
        viewModelScope.launch {
            try {
                val file = File(javaAppManager.workingDir, fileName)
                file.writeText("")
                Timber.i("Created workspace file: ${file.name}")
                refresh()
            } catch (e: Exception) {
                Timber.e(e, "Failed to create workspace file")
            }
        }
    }

    fun deleteWorkspaceFile(file: File) {
        viewModelScope.launch {
            try {
                if (file.delete()) {
                    Timber.i("Deleted workspace file: ${file.name}")
                    refresh()
                }
            } catch (e: Exception) {
                Timber.e(e, "Failed to delete workspace file")
            }
        }
    }

    fun shareFile(context: Context, file: File) {
        viewModelScope.launch {
            try {
                val uri = FileProvider.getUriForFile(
                    context,
                    "${context.packageName}.fileprovider",
                    file
                )
                val intent = Intent(Intent.ACTION_SEND).apply {
                    type = "*/*"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                val chooser = Intent.createChooser(intent, "Share ${file.name}")
                chooser.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(chooser)
            } catch (e: Exception) {
                Timber.e(e, "Failed to share file")
            }
        }
    }
}
