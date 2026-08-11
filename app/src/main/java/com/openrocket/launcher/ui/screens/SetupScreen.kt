package com.openrocket.launcher.ui.screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Info
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
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
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
import com.openrocket.launcher.engine.JavaAppManager
import com.openrocket.launcher.engine.JavaAppSetupState
import com.openrocket.launcher.ui.viewmodel.SetupViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SetupScreen(
    navController: NavController,
    viewModel: SetupViewModel = viewModel()
) {
    val setupState by viewModel.setupState.collectAsState()
    var jdkUrl by remember { mutableStateOf(JavaAppManager.DEFAULT_JDK_URL) }

    // File picker for local JDK import
    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let {
            viewModel.installFromLocalFile(it)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Setup") },
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
                .padding(paddingValues)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text(
                text = "Environment Setup",
                style = MaterialTheme.typography.headlineLarge,
                color = MaterialTheme.colorScheme.primary
            )

            Text(
                text = "Install OpenJDK to run Java applications on your device. " +
                        "After JDK is installed, you can import JAR files from the Files tab.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.width(8.dp))

            // JDK Setup
            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        text = "JDK Installation",
                        style = MaterialTheme.typography.titleLarge
                    )

                    OutlinedTextField(
                        value = jdkUrl,
                        onValueChange = { jdkUrl = it },
                        label = { Text("JDK Download URL") },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                        enabled = setupState !is JavaAppSetupState.DownloadingJdk
                    )

                    SetupStatusDisplay(setupState)

                    when (setupState) {
                        is JavaAppSetupState.Idle,
                        is JavaAppSetupState.Error -> {
                            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Button(
                                    onClick = { viewModel.startSetup(jdkUrl) },
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Icon(Icons.Filled.Download, contentDescription = null)
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text("Download & Install JDK")
                                }

                                OutlinedButton(
                                    onClick = {
                                        filePickerLauncher.launch(
                                            arrayOf(
                                                "application/x-xz",
                                                "application/x-gzip",
                                                "application/x-tar",
                                                "*/*"
                                            )
                                        )
                                    },
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Icon(Icons.Filled.FolderOpen, contentDescription = null)
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text("Import from Local File")
                                }

                                Text(
                                    text = "Supported formats: .tar.xz, .tar.gz",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        is JavaAppSetupState.Ready -> {
                            Button(
                                onClick = { navController.navigateUp() },
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Icon(Icons.Filled.CheckCircle, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Done")
                            }
                        }
                        else -> {
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
                }
            }

            // Import Instructions
            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        text = "How to Import JAR Files",
                        style = MaterialTheme.typography.titleLarge
                    )

                    Text(
                        text = "1. Go to the Files tab\n" +
                                "2. Tap the + button to import\n" +
                                "3. Select a .jar file from your device\n" +
                                "4. The app will be available on the Home screen",
                        style = MaterialTheme.typography.bodyMedium
                    )

                    Text(
                        text = "Note: Only pure Java applications are supported. " +
                                "Native libraries (.so/.dll) will not work on Android.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@Composable
private fun SetupStatusDisplay(state: JavaAppSetupState) {
    when (state) {
        is JavaAppSetupState.Idle -> {
            Text(
                text = "Ready to install JDK",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        is JavaAppSetupState.Checking -> {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Checking installation...")
            }
        }
        is JavaAppSetupState.DownloadingJdk -> {
            Column {
                Text("Downloading JDK... ${(state.progress * 100).toInt()}%")
                LinearProgressIndicator(
                    progress = { state.progress },
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }
        is JavaAppSetupState.DownloadingApp -> {
            Column {
                Text("Downloading ${state.appName}... ${(state.progress * 100).toInt()}%")
                LinearProgressIndicator(
                    progress = { state.progress },
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }
        is JavaAppSetupState.Extracting -> {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Extracting files... ${(state.progress * 100).toInt()}%")
                if (state.currentEntry.isNotBlank()) {
                    Text(
                        text = state.currentEntry,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1
                    )
                }
                LinearProgressIndicator(
                    progress = { state.progress },
                    modifier = Modifier.fillMaxWidth()
                )
            }
        }
        is JavaAppSetupState.Testing -> {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Testing JDK installation...")
            }
        }
        is JavaAppSetupState.Ready -> {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Filled.CheckCircle,
                        contentDescription = null,
                        tint = com.openrocket.launcher.ui.theme.SuccessGreen
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        "JDK installed successfully",
                        color = com.openrocket.launcher.ui.theme.SuccessGreen
                    )
                }
                if (state.versionInfo.isNotBlank()) {
                    Text(
                        text = "Version: ${state.versionInfo}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
        is JavaAppSetupState.Error -> {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Filled.Error,
                    contentDescription = null,
                    tint = com.openrocket.launcher.ui.theme.ErrorRed
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    state.message,
                    color = com.openrocket.launcher.ui.theme.ErrorRed
                )
            }
        }
    }
}
