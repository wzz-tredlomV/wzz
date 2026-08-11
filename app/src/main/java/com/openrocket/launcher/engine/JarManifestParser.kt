package com.openrocket.launcher.engine

import timber.log.Timber
import java.io.File
import java.util.jar.JarFile

/**
 * Parse JAR manifest to extract Main-Class attribute.
 */
object JarManifestParser {

    /**
     * Extract Main-Class from JAR file MANIFEST.MF.
     * Returns null if not found or JAR is invalid.
     */
    fun getMainClass(jarFile: File): String? {
        return try {
            JarFile(jarFile).use { jar ->
                val manifest = jar.manifest
                if (manifest == null) {
                    Timber.w("No manifest found in ${jarFile.name}")
                    return null
                }
                val mainClass = manifest.mainAttributes.getValue("Main-Class")
                if (mainClass != null) {
                    Timber.i("Found Main-Class in ${jarFile.name}: $mainClass")
                    mainClass
                } else {
                    Timber.w("No Main-Class attribute in ${jarFile.name} manifest")
                    null
                }
            }
        } catch (e: Exception) {
            Timber.e(e, "Failed to parse JAR manifest for ${jarFile.name}")
            null
        }
    }
}
