@file:Suppress("DEPRECATION")
package com.openrocket.launcher.ui.navigation

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.Info
import androidx.compose.material.icons.outlined.List
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.openrocket.launcher.ui.screens.AboutScreen
import com.openrocket.launcher.ui.screens.FileBrowserScreen
import com.openrocket.launcher.ui.screens.HomeScreen
import com.openrocket.launcher.ui.screens.LogViewerScreen
import com.openrocket.launcher.ui.screens.SetupScreen
import com.openrocket.launcher.ui.screens.SettingsScreen
import com.openrocket.launcher.ui.screens.VncScreen

sealed class BottomNavItem(
    val screen: Screen,
    val selectedIcon: ImageVector,
    val unselectedIcon: ImageVector,
    val label: String
) {
    object Home : BottomNavItem(
        Screen.Home,
        Icons.Filled.Home,
        Icons.Outlined.Home,
        "Home"
    )
    object Files : BottomNavItem(
        Screen.FileBrowser,
        Icons.Filled.List,
        Icons.Outlined.List,
        "Files"
    )
    object Logs : BottomNavItem(
        Screen.LogViewer,
        Icons.Filled.List,
        Icons.Outlined.List,
        "Logs"
    )
    object Settings : BottomNavItem(
        Screen.Settings,
        Icons.Filled.Settings,
        Icons.Outlined.Settings,
        "Settings"
    )
    object About : BottomNavItem(
        Screen.About,
        Icons.Filled.Info,
        Icons.Outlined.Info,
        "About"
    )
}

@Composable
fun OpenRocketNavHost() {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination

    val bottomNavItems = listOf(
        BottomNavItem.Home,
        BottomNavItem.Files,
        BottomNavItem.Logs,
        BottomNavItem.Settings,
        BottomNavItem.About
    )

    Scaffold(
        bottomBar = {
            NavigationBar {
                bottomNavItems.forEach { item ->
                    val selected = currentDestination?.hierarchy?.any {
                        it.route == item.screen.route
                    } == true

                    NavigationBarItem(
                        icon = {
                            Icon(
                                imageVector = if (selected) item.selectedIcon else item.unselectedIcon,
                                contentDescription = item.label
                            )
                        },
                        label = { Text(item.label) },
                        selected = selected,
                        onClick = {
                            navController.navigate(item.screen.route) {
                                popUpTo(navController.graph.findStartDestination().id) {
                                    saveState = true
                                }
                                launchSingleTop = true
                                restoreState = true
                            }
                        }
                    )
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Home.route,
            modifier = Modifier.padding(innerPadding)
        ) {
            composable(
                route = "home?appName={appName}",
                arguments = listOf(
                    androidx.navigation.navArgument("appName") {
                        type = androidx.navigation.NavType.StringType
                        defaultValue = ""
                    }
                )
            ) { backStackEntry ->
                val appName = backStackEntry.arguments?.getString("appName") ?: ""
                HomeScreen(navController, preselectedApp = appName.takeIf { it.isNotEmpty() })
            }
            composable(Screen.Setup.route) { SetupScreen(navController) }
            composable(Screen.FileBrowser.route) { FileBrowserScreen(navController) }
            composable(Screen.LogViewer.route) { LogViewerScreen() }
            composable(Screen.Settings.route) { SettingsScreen() }
            composable(Screen.About.route) { AboutScreen() }
            composable(Screen.Vnc.route) { VncScreen(navController) }
        }
    }
}
