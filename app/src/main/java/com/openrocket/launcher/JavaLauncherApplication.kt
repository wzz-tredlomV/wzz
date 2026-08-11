package com.openrocket.launcher

import android.app.Application
import com.openrocket.launcher.BuildConfig
import timber.log.Timber

class JavaLauncherApplication : Application() {

    override fun onCreate() {
        super.onCreate()

        if (BuildConfig.DEBUG) {
            Timber.plant(Timber.DebugTree())
        } else {
            Timber.plant(CrashReportingTree())
        }

        Timber.i("Java Launcher initialized")
    }

    private class CrashReportingTree : Timber.Tree() {
        override fun log(priority: Int, tag: String?, message: String, t: Throwable?) {
            android.util.Log.println(priority, tag ?: "JavaLauncher", message)
        }
    }
}
