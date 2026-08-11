package com.openrocket.launcher.ui.viewmodel

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.openrocket.launcher.engine.JavaAppManager
import com.openrocket.launcher.engine.JavaAppSetupState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import timber.log.Timber

class SetupViewModel(application: Application) : AndroidViewModel(application) {

    private val javaAppManager = JavaAppManager(application)

    private val _setupState = MutableStateFlow<JavaAppSetupState>(JavaAppSetupState.Idle)
    val setupState: StateFlow<JavaAppSetupState> = _setupState.asStateFlow()

    init {
        viewModelScope.launch {
            javaAppManager.setupState.collect { state ->
                _setupState.value = state
            }
        }
    }

    fun startSetup(jdkUrl: String) {
        viewModelScope.launch {
            try {
                Timber.i("Starting JDK setup: $jdkUrl")
                val result = javaAppManager.setupJdk(jdkUrl)

                result.fold(
                    onSuccess = {
                        Timber.i("JDK setup completed successfully")
                    },
                    onFailure = { error ->
                        Timber.e(error, "JDK setup failed")
                        _setupState.value = JavaAppSetupState.Error(
                            error.message ?: "JDK setup failed"
                        )
                    }
                )
            } catch (e: Exception) {
                Timber.e(e, "Unexpected error during setup")
                _setupState.value = JavaAppSetupState.Error(e.message ?: "Unexpected error")
            }
        }
    }

    fun downloadApp(appName: String, jarUrl: String) {
        viewModelScope.launch {
            try {
                Timber.i("Downloading app: $appName from $jarUrl")
                val result = javaAppManager.downloadApp(appName, jarUrl)

                result.fold(
                    onSuccess = { file: java.io.File ->
                        Timber.i("App $appName downloaded: ${file.absolutePath}")
                    },
                    onFailure = { error: Throwable ->
                        Timber.e(error, "App download failed: ${error.message}")
                        _setupState.value = JavaAppSetupState.Error(
                            error.message ?: "App download failed"
                        )
                    }
                )
            } catch (e: Exception) {
                Timber.e(e, "Unexpected error downloading app")
                _setupState.value = JavaAppSetupState.Error(e.message ?: "Unexpected error")
            }
        }
    }

    fun checkExistingSetup() {
        viewModelScope.launch {
            javaAppManager.checkSetup()
        }
    }

    fun installFromLocalFile(uri: Uri) {
        viewModelScope.launch {
            try {
                Timber.i("Installing JDK from local file: $uri")
                val result = javaAppManager.installJdkFromLocalFile(uri)

                result.fold(
                    onSuccess = {
                        Timber.i("JDK installed from local file successfully")
                    },
                    onFailure = { error ->
                        Timber.e(error, "Local JDK installation failed")
                        _setupState.value = JavaAppSetupState.Error(
                            error.message ?: "Local JDK installation failed"
                        )
                    }
                )
            } catch (e: Exception) {
                Timber.e(e, "Unexpected error during local JDK installation")
                _setupState.value = JavaAppSetupState.Error(e.message ?: "Unexpected error")
            }
        }
    }
}
