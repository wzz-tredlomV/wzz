package com.openrocket.launcher.vnc

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.PorterDuff
import android.util.AttributeSet
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withContext
import timber.log.Timber

/**
 * VNC SurfaceView for rendering Java GUI applications
 *
 * Displays the framebuffer received from the VNC server and handles
 * touch input forwarding (touch -> mouse, gestures -> scroll/zoom).
 *
 * Usage:
 *   val vncView = findViewById<VncView>(R.id.vnc_view)
 *   vncView.connect("127.0.0.1", 5901)
 */
class VncView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : SurfaceView(context, attrs, defStyleAttr), SurfaceHolder.Callback {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private val vncClient = VncClient()
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)

    private var isConnected = false
    private var serverHost: String = "127.0.0.1"
    private var serverPort: Int = 5901

    // Touch tracking for gesture recognition
    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var isDragging = false

    init {
        holder.setFormat(PixelFormat.RGBA_8888)
        holder.addCallback(this)
        isFocusable = true
        isFocusableInTouchMode = true
    }

    /**
     * Connect to VNC server
     */
    fun connect(host: String, port: Int) {
        serverHost = host
        serverPort = port

        scope.launch {
            val surface = holder.surface
            if (surface.isValid) {
                val success = withContext(Dispatchers.IO) {
                    vncClient.connect(host, port, surface)
                }
                isConnected = success
                if (success) {
                    startFrameUpdateLoop()
                }
            } else {
                Timber.w("Surface not ready, connection deferred")
            }
        }
    }

    /**
     * Disconnect from VNC server
     */
    fun disconnect() {
        isConnected = false
        vncClient.disconnect()
    }

    /**
     * Start requesting framebuffer updates
     */
    private fun startFrameUpdateLoop() {
        scope.launch(Dispatchers.IO) {
            while (isActive && isConnected) {
                try {
                    vncClient.requestUpdate(0, 0, width, height, true)
                    delay(33) // ~30fps
                } catch (e: Exception) {
                    Timber.e(e, "Frame update error")
                    break
                }
            }
        }
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        Timber.d("VNC Surface created")
        // If connect was called before surface was ready, retry now
        if (!isConnected && serverPort > 0) {
            connect(serverHost, serverPort)
        }
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        Timber.d("VNC Surface changed: ${width}x${height}")
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        Timber.d("VNC Surface destroyed")
        disconnect()
    }

    /**
     * Handle touch events - convert to VNC pointer events
     */
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (!isConnected) return false

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastTouchX = event.x
                lastTouchY = event.y
                isDragging = false
                vncClient.sendPointerEvent(event.x.toInt(), event.y.toInt(), 1) // Left click down
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                val dx = event.x - lastTouchX
                val dy = event.y - lastTouchY
                if (kotlin.math.abs(dx) > 5 || kotlin.math.abs(dy) > 5) {
                    isDragging = true
                }
                vncClient.sendPointerEvent(event.x.toInt(), event.y.toInt(), 1)
                lastTouchX = event.x
                lastTouchY = event.y
                return true
            }
            MotionEvent.ACTION_UP -> {
                if (!isDragging) {
                    // It was a tap, send complete click sequence
                    vncClient.sendPointerEvent(event.x.toInt(), event.y.toInt(), 1)
                    vncClient.sendPointerEvent(event.x.toInt(), event.y.toInt(), 0)
                } else {
                    // Drag release
                    vncClient.sendPointerEvent(event.x.toInt(), event.y.toInt(), 0)
                }
                isDragging = false
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    /**
     * Handle keyboard events - forward to VNC server
     */
    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (!isConnected) return super.onKeyDown(keyCode, event)
        event?.let {
            val vncKey = androidKeyToVncKey(keyCode)
            vncClient.sendKeyEvent(vncKey, true)
        }
        return true
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent?): Boolean {
        if (!isConnected) return super.onKeyUp(keyCode, event)
        event?.let {
            val vncKey = androidKeyToVncKey(keyCode)
            vncClient.sendKeyEvent(vncKey, false)
        }
        return true
    }

    /**
     * Convert Android keycode to VNC keysym
     */
    private fun androidKeyToVncKey(keyCode: Int): Int {
        return when (keyCode) {
            KeyEvent.KEYCODE_A -> 0x0061
            KeyEvent.KEYCODE_B -> 0x0062
            KeyEvent.KEYCODE_C -> 0x0063
            KeyEvent.KEYCODE_D -> 0x0064
            KeyEvent.KEYCODE_E -> 0x0065
            KeyEvent.KEYCODE_F -> 0x0066
            KeyEvent.KEYCODE_G -> 0x0067
            KeyEvent.KEYCODE_H -> 0x0068
            KeyEvent.KEYCODE_I -> 0x0069
            KeyEvent.KEYCODE_J -> 0x006A
            KeyEvent.KEYCODE_K -> 0x006B
            KeyEvent.KEYCODE_L -> 0x006C
            KeyEvent.KEYCODE_M -> 0x006D
            KeyEvent.KEYCODE_N -> 0x006E
            KeyEvent.KEYCODE_O -> 0x006F
            KeyEvent.KEYCODE_P -> 0x0070
            KeyEvent.KEYCODE_Q -> 0x0071
            KeyEvent.KEYCODE_R -> 0x0072
            KeyEvent.KEYCODE_S -> 0x0073
            KeyEvent.KEYCODE_T -> 0x0074
            KeyEvent.KEYCODE_U -> 0x0075
            KeyEvent.KEYCODE_V -> 0x0076
            KeyEvent.KEYCODE_W -> 0x0077
            KeyEvent.KEYCODE_X -> 0x0078
            KeyEvent.KEYCODE_Y -> 0x0079
            KeyEvent.KEYCODE_Z -> 0x007A
            KeyEvent.KEYCODE_0 -> 0x0030
            KeyEvent.KEYCODE_1 -> 0x0031
            KeyEvent.KEYCODE_2 -> 0x0032
            KeyEvent.KEYCODE_3 -> 0x0033
            KeyEvent.KEYCODE_4 -> 0x0034
            KeyEvent.KEYCODE_5 -> 0x0035
            KeyEvent.KEYCODE_6 -> 0x0036
            KeyEvent.KEYCODE_7 -> 0x0037
            KeyEvent.KEYCODE_8 -> 0x0038
            KeyEvent.KEYCODE_9 -> 0x0039
            KeyEvent.KEYCODE_SPACE -> 0x0020
            KeyEvent.KEYCODE_ENTER -> 0xFF0D
            KeyEvent.KEYCODE_DEL -> 0xFF08
            KeyEvent.KEYCODE_TAB -> 0xFF09
            KeyEvent.KEYCODE_ESCAPE -> 0xFF1B
            KeyEvent.KEYCODE_DPAD_UP -> 0xFF52
            KeyEvent.KEYCODE_DPAD_DOWN -> 0xFF54
            KeyEvent.KEYCODE_DPAD_LEFT -> 0xFF51
            KeyEvent.KEYCODE_DPAD_RIGHT -> 0xFF53
            KeyEvent.KEYCODE_SHIFT_LEFT -> 0xFFE1
            KeyEvent.KEYCODE_SHIFT_RIGHT -> 0xFFE2
            KeyEvent.KEYCODE_CTRL_LEFT -> 0xFFE3
            KeyEvent.KEYCODE_CTRL_RIGHT -> 0xFFE4
            KeyEvent.KEYCODE_ALT_LEFT -> 0xFFE9
            KeyEvent.KEYCODE_ALT_RIGHT -> 0xFFEA
            else -> keyCode
        }
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        disconnect()
        scope.cancel()
    }
}
