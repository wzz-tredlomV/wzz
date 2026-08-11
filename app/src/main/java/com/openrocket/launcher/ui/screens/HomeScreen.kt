package com.openrocket.launcher.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.DesktopWindows
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Terminal
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.openrocket.launcher.engine.InstalledApp
import com.openrocket.launcher.engine.JavaAppSetupState
import com.openrocket.launcher.engine.ProcessState
import com.openrocket.launcher.ui.navigation.Screen
import com.openrocket.launcher.ui.theme.AccentOrange
import com.openrocket.launcher.ui.theme.ErrorRed
import com.openrocket.launcher.ui.theme.SuccessGreen
import com.openrocket.launcher.ui.viewmodel.HomeViewModel

@Composable
fun HomeScreen(
    navController: NavController,
    preselectedApp: String? = null,
    viewModel: HomeViewModel = viewModel()
) {
    val setupState by viewModel.setupState.collectAsState()
    val processState by viewModel.processState.collectAsState()
    val jdkVersion by viewModel.jdkVersion.collectAsState()
    val installedApps by viewModel.installedApps.collectAsState()
    val isRunning by viewModel.isRunning.collectAsState()
    val selectedApp by viewModel.selectedApp.collectAsState()

    LaunchedEffect(Unit) {
        viewModel.checkSetup()
    }

    LaunchedEffect(preselectedApp) {
        preselectedApp?.let { appName ->
            viewModel.selectApp(appName)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "Java Launcher",
            style = MaterialTheme.typography.displayMedium,
            color = MaterialTheme.colorScheme.primary
        )
        Text(
            text = "Run Java applications on Android",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(8.dp))

        StatusCard(
            setupState = setupState,
            jdkVersion = jdkVersion,
            onSetupClick = { navController.navigate(Screen.Setup.route) }
        )

        if (installedApps.isNotEmpty()) {
            AppSelectorCard(
                apps = installedApps,
                selectedApp = selectedApp,
                onSelectApp = { viewModel.selectApp(it) }
            )
        }

        var useGuiMode by remember { mutableStateOf(false) }
        var showErrorDetails by remember { mutableStateOf(false) }
        val errorMessage = (processState as? ProcessState.Error)?.message ?: ""

        ProcessControlCard(
            isRunning = isRunning,
            processState = processState,
            selectedApp = selectedApp,
            useGuiMode = useGuiMode,
            onUseGuiModeChange = { useGuiMode = it },
            onStart = { viewModel.startJavaApp(useGuiMode = useGuiMode) },
            onStop = { viewModel.stopJavaApp() },
            onViewLogs = { navController.navigate(Screen.LogViewer.route) },
            onToggleErrorDetails = { showErrorDetails = !showErrorDetails }
        )

        // Error Details Dialog
        if (showErrorDetails && processState is ProcessState.Error) {
            AlertDialog(
                onDismissRequest = { showErrorDetails = false },
                title = { Text("Error Details") },
                text = {
                    Text(
                        text = errorMessage,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.error
                    )
                },
                confirmButton = {
                    TextButton(onClick = { showErrorDetails = false }) {
                        Text("OK")
                    }
                }
            )
        }

        QuickActionsCard(
            onBrowseFiles = { navController.navigate(Screen.FileBrowser.route) },
            onSettings = { navController.navigate(Screen.Settings.route) },
            onGuiMode = { navController.navigate(Screen.Vnc.route) }
        )
    }
}

@Composable
private fun StatusCard(
    setupState: JavaAppSetupState,
    jdkVersion: String,
    onSetupClick: () -> Unit
) {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Environment Status",
                style = MaterialTheme.typography.titleLarge
            )

            when (setupState) {
                is JavaAppSetupState.Idle -> {
                    StatusRow("JDK", "Not installed", ErrorRed)
                    Button(onClick = onSetupClick) {
                        Text("Install JDK")
                    }
                }
                is JavaAppSetupState.Checking -> {
                    StatusRow("Status", "Checking...", AccentOrange)
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
                is JavaAppSetupState.DownloadingJdk -> {
                    val progress = setupState.progress
                    StatusRow("JDK", "Downloading...", AccentOrange)
                    LinearProgressIndicator(
                        progress = { progress },
                        modifier = Modifier.fillMaxWidth()
                    )
                    Text("${(progress * 100).toInt()}%", style = MaterialTheme.typography.labelMedium)
                }
                is JavaAppSetupState.DownloadingApp -> {
                    val progress = setupState.progress
                    StatusRow("App", "Downloading ${setupState.appName}...", AccentOrange)
                    LinearProgressIndicator(
                        progress = { progress },
                        modifier = Modifier.fillMaxWidth()
                    )
                    Text("${(progress * 100).toInt()}%", style = MaterialTheme.typography.labelMedium)
                }
                is JavaAppSetupState.Extracting -> {
                    StatusRow("Status", "Extracting... ${(setupState.progress * 100).toInt()}%", AccentOrange)
                    LinearProgressIndicator(
                        progress = { setupState.progress },
                        modifier = Modifier.fillMaxWidth()
                    )
                    if (setupState.currentEntry.isNotBlank()) {
                        Text(
                            text = setupState.currentEntry,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1
                        )
                    }
                }
                is JavaAppSetupState.Testing -> {
                    StatusRow("Status", "Testing JDK...", AccentOrange)
                    CircularProgressIndicator(modifier = Modifier.size(24.dp))
                }
                is JavaAppSetupState.Ready -> {
                    StatusRow("JDK", jdkVersion, SuccessGreen)
                    StatusRow("Status", "Ready", SuccessGreen)
                }
                is JavaAppSetupState.Error -> {
                    StatusRow("Error", setupState.message, ErrorRed)
                    Button(onClick = onSetupClick) {
                        Text("Retry")
                    }
                }
            }
        }
    }
}

@Composable
private fun AppSelectorCard(
    apps: List<InstalledApp>,
    selectedApp: String?,
    onSelectApp: (String) -> Unit
) {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Installed Applications",
                style = MaterialTheme.typography.titleLarge
            )

            apps.forEach { app ->
                val isSelected = app.name == selectedApp
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = if (isSelected) {
                            MaterialTheme.colorScheme.primaryContainer
                        } else {
                            MaterialTheme.colorScheme.surface
                        }
                    ),
                    onClick = { onSelectApp(app.name) }
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = app.name,
                                style = MaterialTheme.typography.bodyLarge
                            )
                            Text(
                                text = if (app.isValid) "Ready" else "Invalid",
                                style = MaterialTheme.typography.labelMedium,
                                color = if (app.isValid) SuccessGreen else ErrorRed
                            )
                        }
                        if (isSelected) {
                            Icon(
                                imageVector = Icons.Filled.PlayArrow,
                                contentDescription = "Selected",
                                tint = MaterialTheme.colorScheme.primary
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ProcessControlCard(
    isRunning: Boolean,
    processState: ProcessState,
    selectedApp: String?,
    useGuiMode: Boolean,
    onUseGuiModeChange: (Boolean) -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onViewLogs: () -> Unit,
    onToggleErrorDetails: () -> Unit
) {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text(
                text = "Process Control",
                style = MaterialTheme.typography.titleLarge
            )

            selectedApp?.let {
                Text(
                    text = "Selected: $it",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.primary
                )
            }

            // GUI Mode toggle
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = if (useGuiMode) Icons.Filled.DesktopWindows else Icons.Filled.Terminal,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp),
                        tint = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Column {
                        Text(
                            text = if (useGuiMode) "GUI Mode" else "Headless Mode",
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            text = if (useGuiMode) "Requires VNC server" else "Command-line only",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
                Switch(
                    checked = useGuiMode,
                    onCheckedChange = onUseGuiModeChange
                )
            }

            val (statusText, statusColor) = when (processState) {
                is ProcessState.Idle -> "Idle" to MaterialTheme.colorScheme.onSurfaceVariant
                is ProcessState.Starting -> "Starting..." to AccentOrange
                is ProcessState.Running -> "Running" to SuccessGreen
                is ProcessState.Stopping -> "Stopping..." to AccentOrange
                is ProcessState.Exited -> {
                    val code = processState.exitCode
                    "Exited (code: $code)" to if (code == 0) SuccessGreen else ErrorRed
                }
                is ProcessState.Error -> "Error" to ErrorRed
            }

            StatusRow("Status", statusText, statusColor)

            // Show error details button when in error state
            if (processState is ProcessState.Error) {
                TextButton(
                    onClick = onToggleErrorDetails,
                    modifier = Modifier.align(Alignment.End)
                ) {
                    Text("View Error Details")
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Button(
                    onClick = onStart,
                    enabled = !isRunning && selectedApp != null,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Filled.PlayArrow, contentDescription = null)
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Start")
                }

                OutlinedButton(
                    onClick = onStop,
                    enabled = isRunning,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Filled.Stop, contentDescription = null)
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Stop")
                }
            }

            OutlinedButton(
                onClick = onViewLogs,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(Icons.Filled.Visibility, contentDescription = null)
                Spacer(modifier = Modifier.width(4.dp))
                Text("View Logs")
            }
        }
    }
}

@Composable
private fun QuickActionsCard(
    onBrowseFiles: () -> Unit,
    onSettings: () -> Unit,
    onGuiMode: () -> Unit
) {
    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Quick Actions",
                style = MaterialTheme.typography.titleLarge
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                OutlinedButton(
                    onClick = onBrowseFiles,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Filled.Add, contentDescription = null)
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Import")
                }

                OutlinedButton(
                    onClick = onGuiMode,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Filled.Visibility, contentDescription = null)
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("GUI")
                }
            }

            OutlinedButton(
                onClick = onSettings,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(Icons.Filled.Settings, contentDescription = null)
                Spacer(modifier = Modifier.width(4.dp))
                Text("Settings")
            }
        }
    }
}

@Composable
private fun StatusRow(
    label: String,
    value: String,
    color: androidx.compose.ui.graphics.Color
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            color = color
        )
    }
}
