package com.openrocket.launcher.vnc

import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.IOException
import java.net.InetSocketAddress
import java.net.Socket
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Pure Kotlin implementation of a VNC client (RFB protocol 3.8).
 * No JNI dependencies. Supports Raw and CopyRect encodings.
 * Renders to an Android Bitmap.
 */
class VncClient {

    private var socket: Socket? = null
    private var input: DataInputStream? = null
    private var output: DataOutputStream? = null

    private var framebufferWidth = 0
    private var framebufferHeight = 0
    private var bitsPerPixel = 0

    private var framebuffer: Bitmap? = null
    private var framebufferPixels: IntArray? = null

    @Volatile
    private var connected = false

    @Volatile
    private var running = false

    companion object {
        private const val RFB_VERSION = "RFB 003.008\n"
        private const val SECURITY_NONE = 1
        private const val MSG_FRAMEBUFFER_UPDATE = 0
        private const val CLIENT_SET_PIXEL_FORMAT = 0
        private const val CLIENT_SET_ENCODINGS = 2
        private const val CLIENT_FRAMEBUFFER_UPDATE_REQUEST = 3
        private const val CLIENT_POINTER_EVENT = 5
        private const val CLIENT_KEY_EVENT = 4
        private const val ENCODING_RAW = 0
        private const val ENCODING_COPYRECT = 1
    }

    suspend fun connect(host: String, port: Int): Boolean = withContext(Dispatchers.IO) {
        try {
            Timber.i("Connecting to VNC server at $host:$port")
            socket = Socket().apply {
                connect(InetSocketAddress(host, port), 10000)
                tcpNoDelay = true
            }
            input = DataInputStream(socket!!.getInputStream())
            output = DataOutputStream(socket!!.getOutputStream())

            if (!handshake()) {
                disconnect()
                return@withContext false
            }

            connected = true
            running = true
            Timber.i("VNC connected, framebuffer: ${framebufferWidth}x$framebufferHeight")
            launchUpdateLoop()
            true
        } catch (e: Exception) {
            Timber.e(e, "VNC connection failed")
            disconnect()
            false
        }
    }

    fun disconnect() {
        running = false
        connected = false
        try { input?.close() } catch (_: Exception) {}
        try { output?.close() } catch (_: Exception) {}
        try { socket?.close() } catch (_: Exception) {}
        input = null
        output = null
        socket = null
    }

    fun isConnected(): Boolean = connected && socket?.isConnected == true
    fun getFramebuffer(): Bitmap? = framebuffer
    fun getFramebufferSize(): Pair<Int, Int> = framebufferWidth to framebufferHeight

    suspend fun requestUpdate(incremental: Boolean = true) = withContext(Dispatchers.IO) {
        try {
            output?.apply {
                writeByte(CLIENT_FRAMEBUFFER_UPDATE_REQUEST)
                writeByte(if (incremental) 1 else 0)
                writeShort(0)
                writeShort(0)
                writeShort(framebufferWidth)
                writeShort(framebufferHeight)
                flush()
            }
        } catch (e: Exception) {
            Timber.w(e, "Failed to send update request")
        }
    }

    suspend fun sendPointerEvent(x: Int, y: Int, buttonMask: Int) = withContext(Dispatchers.IO) {
        try {
            val clampedX = x.coerceIn(0, framebufferWidth - 1)
            val clampedY = y.coerceIn(0, framebufferHeight - 1)
            output?.apply {
                writeByte(CLIENT_POINTER_EVENT)
                writeByte(buttonMask)
                writeShort(clampedX)
                writeShort(clampedY)
                flush()
            }
        } catch (e: Exception) {
            Timber.w(e, "Failed to send pointer event")
        }
    }

    suspend fun sendKeyEvent(keySym: Int, down: Boolean) = withContext(Dispatchers.IO) {
        try {
            output?.apply {
                writeByte(CLIENT_KEY_EVENT)
                writeByte(if (down) 1 else 0)
                writeShort(0)
                writeInt(keySym)
                flush()
            }
        } catch (e: Exception) {
            Timber.w(e, "Failed to send key event")
        }
    }

    private suspend fun handshake(): Boolean = withContext(Dispatchers.IO) {
        try {
            val versionBytes = ByteArray(12)
            input!!.readFully(versionBytes)
            output!!.writeBytes(RFB_VERSION)
            output!!.flush()

            val numSecurityTypes = input!!.readUnsignedByte()
            if (numSecurityTypes == 0) {
                val reasonLen = input!!.readInt()
                val reasonBytes = ByteArray(reasonLen)
                input!!.readFully(reasonBytes)
                Timber.e("Server rejected: ${String(reasonBytes)}")
                return@withContext false
            }

            val securityTypes = ByteArray(numSecurityTypes)
            input!!.readFully(securityTypes)

            if (securityTypes.contains(SECURITY_NONE.toByte())) {
                output!!.writeByte(SECURITY_NONE)
                output!!.flush()
            } else {
                Timber.e("Server does not support None security")
                return@withContext false
            }

            val securityResult = input!!.readInt()
            if (securityResult != 0) {
                val reasonLen = input!!.readInt()
                val reasonBytes = ByteArray(reasonLen)
                input!!.readFully(reasonBytes)
                Timber.e("Security failed: ${String(reasonBytes)}")
                return@withContext false
            }

            output!!.writeByte(1)
            output!!.flush()

            framebufferWidth = input!!.readUnsignedShort()
            framebufferHeight = input!!.readUnsignedShort()
            bitsPerPixel = input!!.readUnsignedByte()
            input!!.readByte() // depth
            input!!.readByte() // bigEndian
            input!!.readByte() // trueColor
            input!!.readUnsignedShort() // redMax
            input!!.readUnsignedShort() // greenMax
            input!!.readUnsignedShort() // blueMax
            input!!.readByte() // redShift
            input!!.readByte() // greenShift
            input!!.readByte() // blueShift
            input!!.readFully(ByteArray(3)) // padding
            val nameLength = input!!.readInt()
            val nameBytes = ByteArray(nameLength)
            input!!.readFully(nameBytes)

            setPixelFormat()
            setEncodings()
            createFramebuffer()
            true
        } catch (e: Exception) {
            Timber.e(e, "Handshake failed")
            false
        }
    }

    private fun setPixelFormat() {
        output!!.apply {
            writeByte(CLIENT_SET_PIXEL_FORMAT)
            writeByte(0)
            writeByte(0)
            writeByte(0)
            writeByte(32)
            writeByte(24)
            writeByte(0)
            writeByte(1)
            writeShort(255)
            writeShort(255)
            writeShort(255)
            writeByte(16)
            writeByte(8)
            writeByte(0)
            writeByte(0)
            writeByte(0)
            writeByte(0)
            flush()
        }
        bitsPerPixel = 32
    }

    private fun setEncodings() {
        output!!.apply {
            writeByte(CLIENT_SET_ENCODINGS)
            writeByte(0)
            writeShort(2)
            writeInt(ENCODING_COPYRECT)
            writeInt(ENCODING_RAW)
            flush()
        }
    }

    private fun createFramebuffer() {
        framebuffer = Bitmap.createBitmap(framebufferWidth, framebufferHeight, Bitmap.Config.ARGB_8888)
        framebufferPixels = IntArray(framebufferWidth * framebufferHeight)
    }

    private suspend fun launchUpdateLoop() {
        withContext(Dispatchers.IO) {
            try {
                requestUpdate(incremental = false)
                while (running && isActive) {
                    if (!readServerMessage()) break
                }
            } catch (e: Exception) {
                if (running) Timber.e(e, "Update loop error")
            } finally {
                connected = false
            }
        }
    }

    private fun readServerMessage(): Boolean {
        return try {
            when (val msgType = input!!.readUnsignedByte()) {
                MSG_FRAMEBUFFER_UPDATE -> handleFramebufferUpdate()
                1 -> skipColorMapEntries()
                2 -> { Timber.d("Server bell"); true }
                3 -> skipServerCutText()
                else -> { Timber.w("Unknown message type: $msgType"); true }
            }
        } catch (e: IOException) {
            if (running) Timber.e(e, "Error reading server message")
            false
        }
    }

    private fun handleFramebufferUpdate(): Boolean {
        return try {
            input!!.readByte()
            val numRects = input!!.readUnsignedShort()
            for (i in 0 until numRects) {
                val x = input!!.readUnsignedShort()
                val y = input!!.readUnsignedShort()
                val width = input!!.readUnsignedShort()
                val height = input!!.readUnsignedShort()
                val encodingType = input!!.readInt()
                when (encodingType) {
                    ENCODING_RAW -> readRawRect(x, y, width, height)
                    ENCODING_COPYRECT -> readCopyRect(x, y, width, height)
                    else -> return false
                }
            }
            if (running) {
                kotlinx.coroutines.runBlocking {
                    delay(50)
                    requestUpdate(incremental = true)
                }
            }
            true
        } catch (e: Exception) {
            Timber.e(e, "Error handling framebuffer update")
            false
        }
    }

    private fun readRawRect(x: Int, y: Int, width: Int, height: Int) {
        val pixels = framebufferPixels ?: return
        val fbWidth = framebufferWidth
        val bytesPerPixel = 4
        val rowBytes = width * bytesPerPixel
        val buffer = ByteArray(rowBytes)
        for (row in 0 until height) {
            input!!.readFully(buffer)
            val byteBuffer = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN)
            for (col in 0 until width) {
                val pixel = byteBuffer.getInt(col * 4)
                val b = (pixel shr 0) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val r = (pixel shr 16) and 0xFF
                val argb = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                val destIndex = (y + row) * fbWidth + (x + col)
                if (destIndex in pixels.indices) pixels[destIndex] = argb
            }
        }
        framebuffer?.setPixels(pixels, 0, fbWidth, 0, 0, fbWidth, framebufferHeight)
    }

    private fun readCopyRect(x: Int, y: Int, width: Int, height: Int) {
        val srcX = input!!.readUnsignedShort()
        val srcY = input!!.readUnsignedShort()
        val pixels = framebufferPixels ?: return
        val fbWidth = framebufferWidth
        for (row in 0 until height) {
            for (col in 0 until width) {
                val srcIndex = (srcY + row) * fbWidth + (srcX + col)
                val destIndex = (y + row) * fbWidth + (x + col)
                if (srcIndex in pixels.indices && destIndex in pixels.indices) {
                    pixels[destIndex] = pixels[srcIndex]
                }
            }
        }
        framebuffer?.setPixels(pixels, 0, fbWidth, 0, 0, fbWidth, framebufferHeight)
    }

    private fun skipColorMapEntries(): Boolean {
        return try {
            input!!.readByte()
            val firstColor = input!!.readUnsignedShort()
            val numColors = input!!.readUnsignedShort()
            for (i in 0 until numColors) {
                input!!.readShort()
                input!!.readShort()
                input!!.readShort()
            }
            true
        } catch (e: Exception) {
            Timber.e(e, "Error skipping color map entries")
            false
        }
    }

    private fun skipServerCutText(): Boolean {
        return try {
            input!!.readByte()
            input!!.readByte()
            input!!.readByte()
            val length = input!!.readInt()
            input!!.readFully(ByteArray(length))
            true
        } catch (e: Exception) {
            Timber.e(e, "Error skipping server cut text")
            false
        }
    }
}
