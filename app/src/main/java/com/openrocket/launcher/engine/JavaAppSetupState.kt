package com.openrocket.launcher.engine

sealed class JavaAppSetupState {
    object Idle : JavaAppSetupState()
    object Checking : JavaAppSetupState()
    data class DownloadingJdk(val progress: Float) : JavaAppSetupState()
    data class DownloadingApp(val progress: Float, val appName: String = "Application") : JavaAppSetupState()
    object Extracting : JavaAppSetupState()
    object Testing : JavaAppSetupState()
    data class Ready(val versionInfo: String = "") : JavaAppSetupState()
    data class Error(val message: String) : JavaAppSetupState()
}
