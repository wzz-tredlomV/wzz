package com.openrocket.launcher.ui.navigation

sealed class Screen(val route: String, val title: String) {
    object Home : Screen("home", "Home")
    object Setup : Screen("setup", "Setup")
    object FileBrowser : Screen("files", "Files")
    object LogViewer : Screen("logs", "Logs")
    object Settings : Screen("settings", "Settings")
    object About : Screen("about", "About")
    object Vnc : Screen("vnc", "VNC Display")

    companion object {
        fun homeWithApp(appName: String): String = "home?appName=${java.net.URLEncoder.encode(appName, "UTF-8")}"
    }
}
