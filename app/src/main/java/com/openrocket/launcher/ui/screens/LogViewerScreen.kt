package com.openrocket.launcher.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.openrocket.launcher.engine.ProcessState
import com.openrocket.launcher.ui.theme.AccentOrange
import com.openrocket.launcher.ui.theme.ErrorRed
import com.openrocket.launcher.ui.theme.LogDebug
import com.openrocket.launcher.ui.theme.LogError
import com.openrocket.launcher.ui.theme.LogInfo
import com.openrocket.launcher.ui.theme.LogWarning
import com.openrocket.launcher.ui.theme.SuccessGreen
import com.openrocket.launcher.ui.viewmodel.LogViewerViewModel
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LogViewerScreen(
    viewModel: LogViewerViewModel = viewModel()
) {
    val logs by viewModel.logs.collectAsState()
    val processState by viewModel.processState.collectAsState()
    val listState = rememberLazyListState()
    val scope = rememberCoroutineScope()
    val clipboardManager = LocalClipboardManager.current
    val context = LocalContext.current

    LaunchedEffect(logs.length) {
        if (logs.isNotEmpty()) {
            scope.launch {
                val lines = logs.lines().size
                if (lines > 0) {
                    listState.animateScrollToItem(lines - 1)
                }
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Process Logs") },
                actions = {
                    IconButton(onClick = {
                        clipboardManager.setText(AnnotatedString(logs))
                    }) {
                        Icon(Icons.Filled.ContentCopy, contentDescription = "Copy logs")
                    }
                    IconButton(onClick = { viewModel.shareLogs(context) }) {
                        Icon(Icons.Filled.Share, contentDescription = "Share logs")
                    }
                    IconButton(onClick = { viewModel.clearLogs() }) {
                        Icon(Icons.Filled.Clear, contentDescription = "Clear logs")
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            ProcessStatusBar(processState)

            if (logs.isEmpty()) {
                EmptyLogsView()
            } else {
                Card(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    ),
                    elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
                ) {
                    SelectionContainer {
                        LazyColumn(
                            state = listState,
                            modifier = Modifier
                                .fillMaxSize()
                                .padding(12.dp),
                            verticalArrangement = Arrangement.spacedBy(2.dp)
                        ) {
                            items(logs.lines()) { line ->
                                LogLineItem(line)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ProcessStatusBar(processState: ProcessState) {
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

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
        colors = CardDefaults.cardColors(
            containerColor = statusColor.copy(alpha = 0.1f)
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(10.dp)
                    .background(
                        color = statusColor,
                        shape = CircleShape
                    )
            )
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text = "Process: $statusText",
                style = MaterialTheme.typography.bodyMedium,
                color = statusColor
            )
        }
    }
}

@Composable
private fun LogLineItem(line: String) {
    val color = when {
        line.contains("ERROR", ignoreCase = true) || line.contains("Exception", ignoreCase = true) ->
            LogError
        line.contains("WARN", ignoreCase = true) ->
            LogWarning
        line.contains("DEBUG", ignoreCase = true) ->
            LogDebug
        line.startsWith(">>>") ->
            AccentOrange
        else -> MaterialTheme.colorScheme.onSurface
    }

    Text(
        text = line,
        style = MaterialTheme.typography.bodySmall,
        color = color,
        fontFamily = FontFamily.Monospace,
        maxLines = 1,
        overflow = TextOverflow.Ellipsis
    )
}

@Composable
private fun EmptyLogsView() {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "No logs yet",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Start a Java application to see process output",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
        )
    }
}
