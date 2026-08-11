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
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.DesktopWindows
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.openrocket.launcher.engine.VncServerConfig
import com.openrocket.launcher.engine.VncServerStatus
import com.openrocket.launcher.ui.theme.ErrorRed
import com.openrocket.launcher.ui.theme.SuccessGreen
import com.openrocket.launcher.ui.viewmodel.VncViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VncScreen(
    navController: NavController,
    viewModel: VncViewModel = viewModel()
) {
    val serverStatus by viewModel.serverStatus.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val serverAvailable by viewModel.serverAvailable.collectAsState()
    val externalHost by viewModel.externalHost.collectAsState()
    val externalPort by viewModel.externalPort.collectAsState()

    var showDownloadDialog by remember { mutableStateOf(false) }
    var downloadUrl by remember { mutableStateOf("") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("GUI Mode") },
                navigationIcon = {
                    IconButton(onClick = { navController.navigateUp() }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(paddingValues)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Header
            Text(
                text = "GUI Display Options",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = "Java GUI applications require an X11 display server. Choose one of the following methods:",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.height(8.dp))

            // === Method 1: Built-in VNC Server ===
            BuiltInVncCard(
                serverAvailable = serverAvailable,
                serverStatus = serverStatus,
                isLoading = isLoading,
                onStartServer = { viewModel.startServer() },
                onStopServer = { viewModel.stopServer() },
                onDownloadClick = { showDownloadDialog = true },
                onRefreshClick = { viewModel.checkBinaries() }
            )

            // === Method 2: External VNC Server ===
            ExternalVncCard(
                host = externalHost,
                port = externalPort,
                onHostChange = { viewModel.setExternalHost(it) },
                onPortChange = { viewModel.setExternalPort(it) },
                onConnect = { /* TODO: launch external VNC viewer intent */ }
            )
        }
    }

    // Download dialog
    if (showDownloadDialog) {
        AlertDialog(
            onDismissRequest = { showDownloadDialog = false },
            title = { Text("Download VNC Binaries") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        "Enter URL for prebuilt Android ARM64 VNC binaries:",
                        style = MaterialTheme.typography.bodySmall
                    )
                    OutlinedTextField(
                        value = downloadUrl,
                        onValueChange = { downloadUrl = it },
                        label = { Text("Download URL") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth()
                    )
                    Text(
                        "Or install Termux and run:\npkg install x11vnc xvfb",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        if (downloadUrl.isNotBlank()) {
                            viewModel.downloadBinaries(downloadUrl)
                            showDownloadDialog = false
                            downloadUrl = ""
                        }
                    },
                    enabled = downloadUrl.isNotBlank() && !isLoading
                ) {
                    Text("Download")
                }
            },
            dismissButton = {
                TextButton(onClick = { showDownloadDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }
}

@Composable
private fun BuiltInVncCard(
    serverAvailable: Boolean,
    serverStatus: VncServerStatus,
    isLoading: Boolean,
    onStartServer: () -> Unit,
    onStopServer: () -> Unit,
    onDownloadClick: () -> Unit,
    onRefreshClick: () -> Unit
) {
    val isRunning = serverStatus is VncServerStatus.Running

    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Title row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Method 1: Built-in VNC Server",
                    style = MaterialTheme.typography.titleLarge
                )
                if (serverAvailable) {
                    Icon(
                        imageVector = Icons.Filled.CheckCircle,
                        contentDescription = "Available",
                        tint = SuccessGreen,
                        modifier = Modifier.size(24.dp)
                    )
                }
            }

            Text(
                text = "One-click start a VNC server inside the app. Uses bundled or system Xvfb + x11vnc binaries. Connect with any VNC viewer app.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.height(4.dp))

            // Status area
            when (serverStatus) {
                is VncServerStatus.Idle -> {
                    if (!serverAvailable) {
                        StatusRow(
                            icon = Icons.Filled.Error,
                            text = "Binaries not available",
                            color = ErrorRed
                        )
                    } else {
                        StatusRow(
                            icon = Icons.Filled.CheckCircle,
                            text = "Ready to start",
                            color = SuccessGreen
                        )
                    }
                }
                is VncServerStatus.Checking -> {
                    StatusRow(
                        icon = null,
                        text = "Checking binaries...",
                        color = MaterialTheme.colorScheme.primary
                    )
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
                is VncServerStatus.Downloading -> {
                    StatusRow(
                        icon = null,
                        text = "Downloading ${serverStatus.component}...",
                        color = MaterialTheme.colorScheme.primary
                    )
                    LinearProgressIndicator(
                        progress = { serverStatus.progress },
                        modifier = Modifier.fillMaxWidth()
                    )
                    Text(
                        "${(serverStatus.progress * 100).toInt()}%",
                        style = MaterialTheme.typography.labelMedium
                    )
                }
                is VncServerStatus.Installing -> {
                    StatusRow(
                        icon = null,
                        text = serverStatus.message,
                        color = MaterialTheme.colorScheme.primary
                    )
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                }
                is VncServerStatus.Starting -> {
                    StatusRow(
                        icon = null,
                        text = "Starting VNC server...",
                        color = MaterialTheme.colorScheme.primary
                    )
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                }
                is VncServerStatus.Running -> {
                    StatusRow(
                        icon = Icons.Filled.CheckCircle,
                        text = "Running on port ${serverStatus.port}",
                        color = SuccessGreen
                    )
                    Text(
                        "Display: ${serverStatus.display}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                is VncServerStatus.Stopping -> {
                    StatusRow(
                        icon = null,
                        text = "Stopping...",
                        color = MaterialTheme.colorScheme.primary
                    )
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                }
                is VncServerStatus.Error -> {
                    StatusRow(
                        icon = Icons.Filled.Error,
                        text = serverStatus.message,
                        color = ErrorRed
                    )
                }
            }

            Spacer(modifier = Modifier.height(4.dp))

            // Action buttons
            if (!serverAvailable) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = onDownloadClick,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(Icons.Filled.CloudDownload, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Download Binaries")
                    }
                    OutlinedButton(
                        onClick = onRefreshClick,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(Icons.Filled.Refresh, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Check Termux / Refresh")
                    }
                }
            } else {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Button(
                        onClick = onStartServer,
                        enabled = !isRunning && !isLoading,
                        modifier = Modifier.weight(1f)
                    ) {
                        Icon(Icons.Filled.PlayArrow, contentDescription = null)
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Start VNC")
                    }

                    OutlinedButton(
                        onClick = onStopServer,
                        enabled = isRunning && !isLoading,
                        modifier = Modifier.weight(1f)
                    ) {
                        Icon(Icons.Filled.Stop, contentDescription = null)
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Stop VNC")
                    }
                }
            }
        }
    }
}

@Composable
private fun ExternalVncCard(
    host: String,
    port: Int,
    onHostChange: (String) -> Unit,
    onPortChange: (Int) -> Unit,
    onConnect: () -> Unit
) {
    var hostText by remember { mutableStateOf(host) }
    var portText by remember { mutableStateOf(port.toString()) }

    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Method 2: External VNC Server",
                style = MaterialTheme.typography.titleLarge
            )

            Text(
                text = "Connect to an existing VNC server running on your device or network.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            OutlinedTextField(
                value = hostText,
                onValueChange = {
                    hostText = it
                    onHostChange(it)
                },
                label = { Text("VNC Host") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth()
            )

            OutlinedTextField(
                value = portText,
                onValueChange = {
                    portText = it
                    it.toIntOrNull()?.let(onPortChange)
                },
                label = { Text("VNC Port") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth()
            )

            Button(
                onClick = onConnect,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(Icons.Filled.DesktopWindows, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Connect")
            }
        }
    }
}

@Composable
private fun StatusRow(
    icon: androidx.compose.ui.graphics.vector.ImageVector?,
    text: String,
    color: androidx.compose.ui.graphics.Color
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        if (icon != null) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = color,
                modifier = Modifier.size(20.dp)
            )
        } else {
            CircularProgressIndicator(
                modifier = Modifier.size(20.dp),
                strokeWidth = 2.dp,
                color = color
            )
        }
        Text(
            text = text,
            style = MaterialTheme.typography.bodyMedium,
            color = color,
            maxLines = 3,
            overflow = TextOverflow.Ellipsis
        )
    }
}
