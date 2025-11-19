package com.rn_cli

import android.content.Intent
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.Promise

class RnGnssModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    init {
        GnssLoggerService.reactContext = reactContext
    }

    override fun getName(): String {
        return "RnGnss"
    }

    @ReactMethod
    fun startLogging(durationSec: Int, promise: Promise) {
        try {
            val intent = Intent(reactApplicationContext, GnssLoggerService::class.java)
            intent.putExtra("DURATION", durationSec)
            reactApplicationContext.startService(intent)
            promise.resolve("GNSS logging started for $durationSec seconds")
        } catch (e: Exception) {
            promise.reject("START_ERROR", "Failed to start GNSS logging", e)
        }
    }

    @ReactMethod
    fun stopLogging(promise: Promise) {
        try {
            val intent = Intent(reactApplicationContext, GnssLoggerService::class.java)
            reactApplicationContext.stopService(intent)
            promise.resolve("GNSS logging stopped")
        } catch (e: Exception) {
            promise.reject("STOP_ERROR", "Failed to stop GNSS logging", e)
        }
    }
}
