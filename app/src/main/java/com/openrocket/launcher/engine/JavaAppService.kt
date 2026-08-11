package com.openrocket.launcher.engine

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.openrocket.launcher.MainActivity
import com.openrocket.launcher.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import timber.log.Timber
import java.io.File

class JavaAppService : Service() {

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private val binder = LocalBinder()
    private lateinit var jarExecutor: JarExecutor
    private lateinit var javaAppManager: JavaAppManager

    inner class LocalBinder : Binder() {
        fun getService(): JavaAppService = this@JavaAppService
    }

    override fun onCreate() {
        super.onCreate()
        jarExecutor = JarExecutor.getInstance(this)
        javaAppManager = JavaAppManager(this)
        createNotificationChannel()
    }

    override fun onBind(intent: Intent): IBinder = binder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val notification = createNotification("Java Launcher running...")
        startForeground(NOTIFICATION_ID, notification)
        return START_NOT_STICKY
    }

    fun getProcessState(): StateFlow<ProcessState> = jarExecutor.processState
    fun getLogs(): StateFlow<String> = jarExecutor.logs

    fun startJavaApp(
        jarFile: File,
        workingDir: File? = null,
        memoryLimit: String = "512m",
        extraJvmArgs: List<String> = emptyList(),
        appArgs: List<String> = emptyList(),
        useVnc: Boolean = false,
        vncDisplay: String = ":1"
    ) {
        serviceScope.launch {
            if (!javaAppManager.checkSetup()) {
                Timber.e("JDK not installed, cannot start Java app")
                return@launch
            }

            val jvmArgs = mutableListOf(
                "-Xms${memoryLimit}",
                "-Xmx${memoryLimit}"
            )

            if (useVnc) {
                // For VNC mode, set display and AWT settings
                jvmArgs.addAll(listOf(
                    "-Djava.awt.headless=false",
                    "-Dawt.useSystemAAFontSettings=lcd",
                    "-Dswing.aatext=true"
                ))
            } else {
                jvmArgs.add("-Djava.awt.headless=true")
            }

            jvmArgs.addAll(extraJvmArgs)

            val javaHome = javaAppManager.javaHome
            val libDir = "$javaHome/lib"
            val libServerDir = "$javaHome/lib/server"
            val ldPath = "$libDir:$libServerDir"

            val envVars = mutableMapOf(
                "JAVA_HOME" to javaHome,
                "PATH" to "$javaHome/bin:${System.getenv("PATH") ?: ""}",
                "LD_LIBRARY_PATH" to ldPath
            )

            // Android 10+ workaround: if running on Android 10+, add additional library paths
            // and try to use linker-friendly settings
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                // Some devices need the full library path including all subdirectories
                val extendedLdPath = buildString {
                    append(ldPath)
                    // Add common library subdirectories
                    listOf("jli", "server", "client", "jvm").forEach { subdir ->
                        val subDirPath = "$libDir/$subdir"
                        if (File(subDirPath).exists()) {
                            append(":$subDirPath")
                        }
                    }
                }
                envVars["LD_LIBRARY_PATH"] = extendedLdPath
                Timber.i("Android 10+ detected, extended LD_LIBRARY_PATH: $extendedLdPath")
            }

            if (useVnc) {
                envVars["DISPLAY"] = vncDisplay
            }

            updateNotification("Java app is running: ${jarFile.name}")

            jarExecutor.startJar(
                javaBinary = javaAppManager.javaBinary,
                jarFile = jarFile,
                workingDir = workingDir,
                jvmArgs = jvmArgs,
                appArgs = appArgs,
                envVars = envVars
            )
        }
    }

    fun stopJavaApp(force: Boolean = false) {
        serviceScope.launch {
            jarExecutor.stopProcess(force)
            updateNotification("Java app stopped")
        }
    }

    fun clearLogs() {
        jarExecutor.clearLogs()
    }

    fun isRunning(): Boolean = jarExecutor.isRunning()

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Java App Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Background service for running Java applications"
            }

            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun createNotification(contentText: String): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Java Launcher for Android")
            .setContentText(contentText)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(contentText: String) {
        val notification = createNotification(contentText)
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
        if (jarExecutor.isRunning()) {
            runBlocking {
                jarExecutor.stopProcess(force = true)
            }
        }
    }

    companion object {
        private const val CHANNEL_ID = "java_app_service"
        private const val NOTIFICATION_ID = 1001
    }
}
