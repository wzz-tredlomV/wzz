package com.openrocket.launcher.ui.screens

import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.openrocket.launcher.ui.viewmodel.JdkInstallState
import com.openrocket.launcher.ui.viewmodel.SettingsViewModel
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    viewModel: SettingsViewModel = viewModel()
) {
    val settings by viewModel.settings.collectAsState()
    val jdkState by viewModel.jdkInstallState.collectAsState()
    val jdkTestLog by viewModel.jdkTestLog.collectAsState()
    var showClearCacheDialog by remember { mutableStateOf(false) }
    var showResetDialog by remember { mutableStateOf(false) }
    var showStorageInfo by remember { mutableStateOf(false) }
    var showTestLog by remember { mutableStateOf(false) }
    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    val context = LocalContext.current

    // File picker for local JDK import
    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let {
            viewModel.installJdkFromLocalFile(it)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Spacer(modifier = Modifier.height(8.dp))

            // JDK Installation Section
            SettingsSection(title = "JDK Installation") {
                JdkInstallCard(
                    jdkState = jdkState,
                    jdkUrl = settings.jdkUrl,
                    onJdkUrlChange = { viewModel.updateJdkUrl(it) },
                    onDownload = { viewModel.installJdk(settings.jdkUrl) },
                    onImportLocal = {
                        filePickerLauncher.launch(
                            arrayOf(
                                "application/x-xz",
                                "application/x-gzip",
                                "application/x-tar",
                                "*/*"
                            )
                        )
                    },
                    onRunTest = { viewModel.runJdkTest() },
                    testLog = jdkTestLog,
                    showTestLog = showTestLog,
                    onShowTestLogChange = { showTestLog = it }
                )
            }

            SettingsSection(title = "JVM Settings") {
                OutlinedTextField(
                    value = settings.memoryLimit,
                    onValueChange = { viewModel.updateMemoryLimit(it) },
                    label = { Text("Max Memory (e.g., 512m, 1g)") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )

                Spacer(modifier = Modifier.height(8.dp))

                OutlinedTextField(
                    value = settings.extraJvmArgs,
                    onValueChange = { viewModel.updateExtraJvmArgs(it) },
                    label = { Text("Extra JVM Arguments") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )
            }

            SettingsSection(title = "Download Sources") {
                OutlinedTextField(
                    value = settings.jdkUrl,
                    onValueChange = { viewModel.updateJdkUrl(it) },
                    label = { Text("JDK Download URL") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )
            }

            SettingsSection(title = "App Settings") {
                SettingsSwitchItem(
                    title = "Auto-start on launch",
                    subtitle = "Automatically start last app when app opens",
                    checked = settings.autoStart,
                    onCheckedChange = { viewModel.updateAutoStart(it) }
                )

                SettingsSwitchItem(
                    title = "Keep screen on",
                    subtitle = "Prevent screen from turning off while running",
                    checked = settings.keepScreenOn,
                    onCheckedChange = { viewModel.updateKeepScreenOn(it) }
                )

                SettingsSwitchItem(
                    title = "Dark theme",
                    subtitle = "Use dark color scheme",
                    checked = settings.darkTheme,
                    onCheckedChange = { viewModel.updateDarkTheme(it) }
                )
            }

            SettingsSection(title = "Maintenance") {
                SettingsActionItem(
                    title = "Clear Cache",
                    subtitle = "Remove temporary downloaded files",
                    icon = Icons.Filled.Delete,
                    onClick = { showClearCacheDialog = true }
                )

                SettingsActionItem(
                    title = "Reset to Defaults",
                    subtitle = "Restore all settings to default values",
                    icon = Icons.Filled.Refresh,
                    onClick = { showResetDialog = true }
                )

                SettingsActionItem(
                    title = "Storage Info",
                    subtitle = "View disk usage statistics",
                    icon = Icons.Filled.Storage,
                    onClick = {
                        viewModel.showStorageInfo()
                        showStorageInfo = true
                    }
                )
            }

            Spacer(modifier = Modifier.height(32.dp))

            Text(
                text = "Java Launcher for Android v1.0.0",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.align(Alignment.CenterHorizontally)
            )
        }
    }

    if (showClearCacheDialog) {
        AlertDialog(
            onDismissRequest = { showClearCacheDialog = false },
            title = { Text("Clear Cache") },
            text = { Text("Delete all temporary downloaded files? JDK and app installations will NOT be affected.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        viewModel.clearCache()
                        showClearCacheDialog = false
                        Toast.makeText(context, "Cache cleared successfully", Toast.LENGTH_SHORT).show()
                    }
                ) {
                    Text("Clear", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { showClearCacheDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }

    if (showResetDialog) {
        AlertDialog(
            onDismissRequest = { showResetDialog = false },
            title = { Text("Reset Settings") },
            text = { Text("Restore all settings to defaults? Your files and installations will NOT be affected.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        viewModel.resetSettings()
                        showResetDialog = false
                        Toast.makeText(context, "Settings reset to defaults", Toast.LENGTH_SHORT).show()
                    }
                ) {
                    Text("Reset", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { showResetDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }

    if (showStorageInfo) {
        val storageInfo by viewModel.storageInfo.collectAsState()
        AlertDialog(
            onDismissRequest = { showStorageInfo = false },
            title = { Text("Storage Info") },
            text = { Text(storageInfo.ifEmpty { "Loading..." }) },
            confirmButton = {
                TextButton(onClick = { showStorageInfo = false }) {
                    Text("OK")
                }
            }
        )
    }
}

@Composable
private fun JdkInstallCard(
    jdkState: JdkInstallState,
    jdkUrl: String,
    onJdkUrlChange: (String) -> Unit,
    onDownload: () -> Unit,
    onImportLocal: () -> Unit,
    onRunTest: () -> Unit,
    testLog: String,
    showTestLog: Boolean,
    onShowTestLogChange: (Boolean) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        OutlinedTextField(
            value = jdkUrl,
            onValueChange = onJdkUrlChange,
            label = { Text("JDK Download URL") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            enabled = jdkState !is JdkInstallState.Downloading
        )

        // Status display
        when (jdkState) {
            is JdkInstallState.Idle -> {
                Text(
                    text = "JDK not installed. Download or import a JDK to get started.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            is JdkInstallState.Checking -> {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Checking installation...")
                }
            }
            is JdkInstallState.Downloading -> {
                Column {
                    Text("Downloading JDK... ${(jdkState.progress * 100).toInt()}%")
                    LinearProgressIndicator(
                        progress = { jdkState.progress },
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }
            is JdkInstallState.Extracting -> {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("Extracting... ${(jdkState.progress * 100).toInt()}%")
                    if (jdkState.currentEntry.isNotBlank()) {
                        Text(
                            text = jdkState.currentEntry,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1
                        )
                    }
                    LinearProgressIndicator(
                        progress = { jdkState.progress },
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }
            is JdkInstallState.Testing -> {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Testing JDK installation...")
                }
            }
            is JdkInstallState.Ready -> {
                var showVersionDetails by remember { mutableStateOf(false) }
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Filled.Info,
                            contentDescription = null,
                            tint = com.openrocket.launcher.ui.theme.SuccessGreen,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "JDK installed successfully",
                            color = com.openrocket.launcher.ui.theme.SuccessGreen,
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                    if (jdkState.versionInfo.isNotBlank()) {
                        Text(
                            text = "Version: ${jdkState.versionInfo}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    OutlinedButton(
                        onClick = { showVersionDetails = !showVersionDetails },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(Icons.Filled.Info, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(if (showVersionDetails) "Hide JDK Details" else "View JDK & Tools Versions")
                    }
                    if (showVersionDetails) {
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surfaceVariant
                            )
                        ) {
                            Text(
                                text = testLog.ifBlank { "Click \"Run JDK Test\" above to load detailed version info." },
                                style = MaterialTheme.typography.bodySmall,
                                modifier = Modifier.padding(12.dp),
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }
            is JdkInstallState.Error -> {
                Text(
                    text = "Error: ${jdkState.message}",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }

        // Action buttons
        when (jdkState) {
            is JdkInstallState.Idle,
            is JdkInstallState.Error,
            is JdkInstallState.Ready -> {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = onDownload,
                        modifier = Modifier.fillMaxWidth(),
                        enabled = jdkState !is JdkInstallState.Downloading
                    ) {
                        Icon(Icons.Filled.CloudDownload, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Download & Install JDK")
                    }

                    OutlinedButton(
                        onClick = onImportLocal,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Icon(Icons.Filled.FolderOpen, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Import from Local File")
                    }

                    if (jdkState is JdkInstallState.Ready) {
                        OutlinedButton(
                            onClick = onRunTest,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Filled.Info, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Run JDK Test")
                        }
                    }
                }
            }
            else -> {
                // Installing state - show disabled button
                Button(
                    onClick = { },
                    enabled = false,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Installing...")
                }
            }
        }

        // Test log display
        if (testLog.isNotBlank()) {
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Test Log",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.primary
                )
                TextButton(onClick = { onShowTestLogChange(!showTestLog) }) {
                    Text(if (showTestLog) "Hide" else "Show")
                }
            }
            if (showTestLog) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Text(
                        text = testLog,
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp),
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@Composable
private fun SettingsSection(
    title: String,
    content: @Composable () -> Unit
) {
    Column {
        Text(
            text = title,
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.padding(vertical = 8.dp)
        )
        Card(
            modifier = Modifier.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                content()
            }
        }
    }
}

@Composable
private fun SettingsSwitchItem(
    title: String,
    subtitle: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(text = title, style = MaterialTheme.typography.bodyLarge)
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange
        )
    }
}

@Composable
private fun SettingsActionItem(
    title: String,
    subtitle: String,
    icon: ImageVector,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(end = 16.dp)
        )
        Column(modifier = Modifier.weight(1f)) {
            Text(text = title, style = MaterialTheme.typography.bodyLarge)
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
