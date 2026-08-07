package com.openrocket.launcher.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.openrocket.launcher.engine.ProcessState
import com.openrocket.launcher.ui.viewmodel.VncConnectionState
import com.openrocket.launcher.ui.viewmodel.VncServerState
import com.openrocket.launcher.ui.viewmodel.VncViewModel
import kotlinx.coroutines.launch
import timber.log.Timber

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VncScreen(
    navController: NavController,
    appName: String? = null,
    viewModel: VncViewModel = viewModel()
) {
    val connectionState by viewModel.connectionState.collectAsState()
    val vncServerState by viewModel.vncServerState.collectAsState()
    val framebuffer by viewModel.framebuffer.collectAsState()
    val processState by viewModel.processState.collectAsState()
    val serverLog by viewModel.serverLog.collectAsState()

    val snackbarHostState = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()
    var showLog by remember { mutableStateOf(false) }
    var scaleToFit by remember { mutableStateOf(true) }

    LaunchedEffect(appName) {
        if (!appName.isNullOrBlank()) {
            Timber.i("Auto-starting GUI session for app: $appName")
            viewModel.startGuiApp(appName)
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            viewModel.stopAll()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("VNC Viewer")
                        Text(
                            text = when (connectionState) {
                                is VncConnectionState.Connected -> "Connected"
                                is VncConnectionState.Connecting -> "Connecting..."
                                is VncConnectionState.Error -> "Error"
                                else -> "Disconnected"
                            },
                            style = MaterialTheme.typography.labelSmall,
                            color = when (connectionState) {
                                is VncConnectionState.Connected -> MaterialTheme.colorScheme.primary
                                is VncConnectionState.Error -> MaterialTheme.colorScheme.error
                                else -> MaterialTheme.colorScheme.onSurfaceVariant
                            }
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = {
                        viewModel.stopAll()
                        navController.navigateUp()
                    }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(onClick = { scaleToFit = !scaleToFit }) {
                        Icon(
                            imageVector = if (scaleToFit) Icons.Filled.Fullscreen else Icons.Filled.FullscreenExit,
                            contentDescription = if (scaleToFit) "Scale to fit" else "1:1"
                        )
                    }
                    IconButton(onClick = { showLog = !showLog }) {
                        Icon(
                            imageVector = if (showLog) Icons.Filled.VisibilityOff else Icons.Filled.Visibility,
                            contentDescription = "Toggle log"
                        )
                    }
                    if (processState is ProcessState.Running || vncServerState is VncServerState.Running) {
                        IconButton(onClick = {
                            viewModel.stopAll()
                            scope.launch {
                                snackbarHostState.showSnackbar("Session stopped")
                            }
                        }) {
                            Icon(
                                imageVector = Icons.Filled.Stop,
                                contentDescription = "Stop",
                                tint = MaterialTheme.colorScheme.error
                            )
                        }
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            ServerStatusBar(vncServerState = vncServerState)

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            ) {
                when {
                    connectionState is VncConnectionState.Error -> {
                        ErrorView(
                            message = (connectionState as VncConnectionState.Error).message,
                            onRetry = {
                                if (!appName.isNullOrBlank()) {
                                    viewModel.startGuiApp(appName)
                                } else {
                                    viewModel.startGuiSession()
                                }
                            }
                        )
                    }
                    connectionState is VncConnectionState.Connecting -> {
                        LoadingView(message = "Connecting to VNC server...")
                    }
                    vncServerState is VncServerState.Starting -> {
                        LoadingView(message = "Starting VNC server...")
                    }
                    framebuffer != null && connectionState is VncConnectionState.Connected -> {
                        VncCanvas(
                            framebuffer = framebuffer!!,
                            scaleToFit = scaleToFit,
                            onPointerEvent = { x, y, buttonMask ->
                                viewModel.sendPointerEvent(x, y, buttonMask)
                            }
                        )
                    }
                    else -> {
                        StartOptionsView(
                            appName = appName,
                            onStartBuiltIn = {
                                if (!appName.isNullOrBlank()) {
                                    viewModel.startGuiApp(appName)
                                } else {
                                    viewModel.startGuiSession()
                                }
                            },
                            onConnectExternal = { host, port ->
                                viewModel.connectToExternalVnc(host, port)
                            }
                        )
                    }
                }
            }

            AnimatedVisibility(visible = showLog) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 200.dp)
                        .padding(8.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Text(
                        text = serverLog.ifBlank { "No log output yet..." },
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp)
                            .verticalScroll(rememberScrollState()),
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@Composable
private fun ServerStatusBar(vncServerState: VncServerState) {
    val (text, color) = when (vncServerState) {
        is VncServerState.NotRunning -> "Server: Stopped" to MaterialTheme.colorScheme.onSurfaceVariant
        is VncServerState.Checking -> "Server: Checking..." to MaterialTheme.colorScheme.primary
        is VncServerState.Starting -> "Server: Starting..." to MaterialTheme.colorScheme.primary
        is VncServerState.Running -> "Server: Running on ${vncServerState.display}:${vncServerState.port}" to MaterialTheme.colorScheme.primary
        is VncServerState.Error -> "Server: Error" to MaterialTheme.colorScheme.error
    }

    Surface(
        color = color.copy(alpha = 0.1f),
        modifier = Modifier.fillMaxWidth()
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.labelMedium,
            color = color,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
        )
    }
}

@Composable
private fun VncCanvas(
    framebuffer: android.graphics.Bitmap,
    scaleToFit: Boolean,
    onPointerEvent: (x: Int, y: Int, buttonMask: Int) -> Unit
) {
    var bitmapOffset by remember { mutableStateOf(Offset.Zero) }
    var bitmapScale by remember { mutableStateOf(1f) }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .pointerInput(scaleToFit) {
                detectTapGestures { offset ->
                    val x = ((offset.x - bitmapOffset.x) / bitmapScale).toInt()
                    val y = ((offset.y - bitmapOffset.y) / bitmapScale).toInt()
                    onPointerEvent(x, y, 1)
                    onPointerEvent(x, y, 0)
                }
            }
            .pointerInput(scaleToFit) {
                detectDragGestures(
                    onDragStart = { offset ->
                        val x = ((offset.x - bitmapOffset.x) / bitmapScale).toInt()
                        val y = ((offset.y - bitmapOffset.y) / bitmapScale).toInt()
                        onPointerEvent(x, y, 1)
                    },
                    onDragEnd = { onPointerEvent(-1, -1, 0) },
                    onDragCancel = { onPointerEvent(-1, -1, 0) },
                    onDrag = { change, _ ->
                        val x = ((change.position.x - bitmapOffset.x) / bitmapScale).toInt()
                        val y = ((change.position.y - bitmapOffset.y) / bitmapScale).toInt()
                        onPointerEvent(x, y, 1)
                    }
                )
            }
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val bmpWidth = framebuffer.width.toFloat()
            val bmpHeight = framebuffer.height.toFloat()

            if (scaleToFit) {
                val scaleX = size.width / bmpWidth
                val scaleY = size.height / bmpHeight
                bitmapScale = minOf(scaleX, scaleY)
                val scaledWidth = bmpWidth * bitmapScale
                val scaledHeight = bmpHeight * bitmapScale
                bitmapOffset = Offset(
                    (size.width - scaledWidth) / 2f,
                    (size.height - scaledHeight) / 2f
                )
            } else {
                bitmapScale = 1f
                bitmapOffset = Offset(
                    (size.width - bmpWidth) / 2f,
                    (size.height - bmpHeight) / 2f
                )
            }

            drawImage(
                image = framebuffer.asImageBitmap(),
                topLeft = bitmapOffset,
                alpha = 1f
            )
        }
    }
}

@Composable
private fun StartOptionsView(
    appName: String?,
    onStartBuiltIn: () -> Unit,
    onConnectExternal: (String, Int) -> Unit
) {
    var externalHost by remember { mutableStateOf("127.0.0.1") }
    var externalPort by remember { mutableStateOf("5901") }
    var showExternal by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = Icons.Filled.DesktopWindows,
            contentDescription = null,
            modifier = Modifier.size(80.dp),
            tint = MaterialTheme.colorScheme.primary.copy(alpha = 0.5f)
        )
        Spacer(modifier = Modifier.height(24.dp))
        Text(
            text = "VNC Viewer",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.primary
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = if (appName != null) "Click below to start $appName with GUI"
            else "Start a built-in VNC server or connect to an external one",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(32.dp))

        Card(
            modifier = Modifier.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text("Built-in VNC Server", style = MaterialTheme.typography.titleMedium)
                Text(
                    "Start Xvfb + x11vnc automatically inside the app.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Button(onClick = onStartBuiltIn, modifier = Modifier.fillMaxWidth()) {
                    Icon(Icons.Filled.PlayArrow, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(if (appName != null) "Start $appName (GUI)" else "Start VNC Server")
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        OutlinedButton(
            onClick = { showExternal = !showExternal },
            modifier = Modifier.fillMaxWidth()
        ) {
            Icon(Icons.Filled.Cloud, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text("Connect to External VNC")
        }

        AnimatedVisibility(visible = showExternal) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 8.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedTextField(
                        value = externalHost,
                        onValueChange = { externalHost = it },
                        label = { Text("Host") },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true
                    )
                    OutlinedTextField(
                        value = externalPort,
                        onValueChange = { externalPort = it },
                        label = { Text("Port") },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true
                    )
                    Button(
                        onClick = {
                            val port = externalPort.toIntOrNull() ?: 5901
                            onConnectExternal(externalHost, port)
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Connect")
                    }
                }
            }
        }
    }
}

@Composable
private fun LoadingView(message: String) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        CircularProgressIndicator(modifier = Modifier.size(48.dp), strokeWidth = 4.dp)
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = message,
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun ErrorView(message: String, onRetry: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = Icons.Filled.Error,
            contentDescription = null,
            modifier = Modifier.size(64.dp),
            tint = MaterialTheme.colorScheme.error
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "Connection Error",
            style = MaterialTheme.typography.titleLarge,
            color = MaterialTheme.colorScheme.error
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = message,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = onRetry) {
            Icon(Icons.Filled.Refresh, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text("Retry")
        }
    }
}
