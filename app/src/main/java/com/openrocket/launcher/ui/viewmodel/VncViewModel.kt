package com.openrocket.launcher.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.openrocket.launcher.engine.VncServerConfig
import com.openrocket.launcher.engine.VncServerManager
import com.openrocket.launcher.engine.VncServerStatus
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import timber.log.Timber

/**
 * ViewModel for the VNC / GUI Mode screen.
 *
 * Bridges the UI with [VncServerManager] and exposes:
 * - Built-in server status (idle / checking / downloading / running / error)
 * - External VNC connection parameters
 * - Loading state for long-running operations
 */
class VncViewModel(application: Application) : AndroidViewModel(application) {

    private val vncServerManager = VncServerManager.getInstance(application)

    /** Expose the manager's state flow directly — single source of truth. */
    val serverStatus: StateFlow<VncServerStatus> = vncServerManager.serverStatus

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _serverAvailable = MutableStateFlow(false)
    val serverAvailable: StateFlow<Boolean> = _serverAvailable.asStateFlow()

    private val _externalHost = MutableStateFlow("localhost")
    val externalHost: StateFlow<String> = _externalHost.asStateFlow()

    private val _externalPort = MutableStateFlow(5900)
    val externalPort: StateFlow<Int> = _externalPort.asStateFlow()

    init {
        viewModelScope.launch {
            checkBinaries()
        }
    }

    // ------------------------------------------------------------------
    // Built-in server controls
    // ------------------------------------------------------------------

    fun checkBinaries() {
        viewModelScope.launch {
            _isLoading.value = true
            _serverAvailable.value = vncServerManager.checkBuiltInBinaries()
            _isLoading.value = false
        }
    }

    fun downloadBinaries(url: String = VncServerManager.DEFAULT_DOWNLOAD_URL) {
        viewModelScope.launch {
            _isLoading.value = true
            val success = vncServerManager.downloadBinaries(url)
            _serverAvailable.value = success
            _isLoading.value = false
        }
    }

    fun startServer(config: VncServerConfig = VncServerConfig()) {
        viewModelScope.launch {
            _isLoading.value = true
            vncServerManager.startBuiltInServer(config)
            _isLoading.value = false
        }
    }

    fun stopServer() {
        viewModelScope.launch {
            vncServerManager.stopServer()
        }
    }

    // ------------------------------------------------------------------
    // External server parameters
    // ------------------------------------------------------------------

    fun setExternalHost(host: String) {
        _externalHost.value = host
    }

    fun setExternalPort(port: Int) {
        _externalPort.value = port
    }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    override fun onCleared() {
        super.onCleared()
        viewModelScope.launch {
            vncServerManager.stopServer()
        }
    }
}
