package com.rn_cli

import androidx.annotation.RequiresApi
import android.app.*
import android.content.Intent
import android.hardware.*
import android.location.*
import android.os.*
import android.util.Log
import android.util.Base64
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.modules.core.DeviceEventManagerModule
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

@RequiresApi(Build.VERSION_CODES.N)
class GnssLoggerService : Service(), SensorEventListener {

    private val TAG = "GnssLoggerService"
    private lateinit var locManager: LocationManager
    private lateinit var sensorManager: SensorManager
    private var outFile: File? = null
    private var running = false
    private val epochBuffer = ConcurrentLinkedQueue<JSONObject>()
    private val imuBuffer = ConcurrentLinkedQueue<JSONObject>()
    private val gnssStatusBuffer = ConcurrentLinkedQueue<JSONObject>()
    private val navMessageBuffer = ConcurrentLinkedQueue<JSONObject>()
    private val bufferLock = ReentrantLock()
    private val handler = Handler(Looper.getMainLooper())
    private var duration: Int = 120
    private var satCount = 0
    private var gnssMeasurementsCallback: GnssMeasurementsEvent.Callback? = null
    private var gnssStatusCallback: GnssStatus.Callback? = null
    private var navMessageCallback: GnssNavigationMessage.Callback? = null

    companion object {
        var reactContext: ReactApplicationContext? = null
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        locManager = getSystemService(LOCATION_SERVICE) as LocationManager
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager
        createNotification()
    }

    private fun createNotification() {
        val chId = "gnss_logger_ch"
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nm.createNotificationChannel(
                NotificationChannel(chId, "GNSS Logger", NotificationManager.IMPORTANCE_LOW)
            )
        }
        val n = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, chId)
                .setContentTitle("Logging GNSS")
                .setContentText("Collecting scientific-grade GNSS & IMU data")
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .build()
        } else {
            Notification.Builder(this)
                .setContentTitle("Logging GNSS")
                .setContentText("Collecting scientific-grade GNSS & IMU data")
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .build()
        }
        startForeground(1, n)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        duration = intent?.getIntExtra("DURATION", 120) ?: 120
        outFile = File(filesDir, "gnss_${System.currentTimeMillis()}.json")
        startLogging()
        handler.postDelayed({ stopLogging() }, (duration * 1000).toLong())
        return START_STICKY
    }

    @Suppress("MissingPermission")
    private fun startLogging() {
        if (running) return
        running = true
        sendEventToRN("gnss_log", "Scientific-grade GNSS logging started")

        gnssMeasurementsCallback = object : GnssMeasurementsEvent.Callback() {
            override fun onGnssMeasurementsReceived(event: GnssMeasurementsEvent) {
                bufferLock.withLock {
                    val epoch = JSONObject()
                    
                    val clock = JSONObject().apply {
                        put("timeNanos", event.clock.timeNanos)
                        put("fullBiasNanos", if (event.clock.hasFullBiasNanos()) event.clock.fullBiasNanos else null)
                        put("biasNanos", if (event.clock.hasBiasNanos()) event.clock.biasNanos else null)
                        put("biasUncertaintyNanos", if (event.clock.hasBiasUncertaintyNanos()) event.clock.biasUncertaintyNanos else null)
                        put("driftNanosPerSecond", if (event.clock.hasDriftNanosPerSecond()) event.clock.driftNanosPerSecond else null)
                        put("driftUncertaintyNanosPerSecond", if (event.clock.hasDriftUncertaintyNanosPerSecond()) event.clock.driftUncertaintyNanosPerSecond else null)
                        put("hardwareClockDiscontinuityCount", event.clock.hardwareClockDiscontinuityCount)
                        put("timeUncertaintyNanos", if (event.clock.hasTimeUncertaintyNanos()) event.clock.timeUncertaintyNanos else null)
                    }
                    epoch.put("clock", clock)
                    
                    val measurements = JSONArray()
                    satCount = event.measurements.size
                    
                    for (m in event.measurements) {
                        val sat = JSONObject().apply {
                            put("svid", m.svid)
                            put("constellationType", m.constellationType)
                            put("timeOffsetNanos", m.timeOffsetNanos)
                            put("state", m.state)
                            put("receivedSvTimeNanos", m.receivedSvTimeNanos)
                            put("receivedSvTimeUncertaintyNanos", m.receivedSvTimeUncertaintyNanos)
                            put("cn0DbHz", m.cn0DbHz)
                            put("pseudorangeRateMetersPerSecond", m.pseudorangeRateMetersPerSecond)
                            put("pseudorangeRateUncertaintyMetersPerSecond", m.pseudorangeRateUncertaintyMetersPerSecond)
                            put("accumulatedDeltaRangeState", m.accumulatedDeltaRangeState)
                            put("accumulatedDeltaRangeMeters", m.accumulatedDeltaRangeMeters)
                            put("accumulatedDeltaRangeUncertaintyMeters", m.accumulatedDeltaRangeUncertaintyMeters)
                            
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                                put("carrierFrequencyHz", if (m.hasCarrierFrequencyHz()) m.carrierFrequencyHz else null)
                                put("multipathIndicator", m.multipathIndicator)
                            }
                            
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                                put("basebandCn0DbHz", if (m.hasBasebandCn0DbHz()) m.basebandCn0DbHz else null)
                            }
                            
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                                put("fullInterSignalBiasNanos", if (m.hasFullInterSignalBiasNanos()) m.fullInterSignalBiasNanos else null)
                                put("fullInterSignalBiasUncertaintyNanos", if (m.hasFullInterSignalBiasUncertaintyNanos()) m.fullInterSignalBiasUncertaintyNanos else null)
                                put("satelliteInterSignalBiasNanos", if (m.hasSatelliteInterSignalBiasNanos()) m.satelliteInterSignalBiasNanos else null)
                                put("satelliteInterSignalBiasUncertaintyNanos", if (m.hasSatelliteInterSignalBiasUncertaintyNanos()) m.satelliteInterSignalBiasUncertaintyNanos else null)
                                put("codeType", if (m.hasCodeType()) m.codeType else null)
                            }
                        }
                        measurements.put(sat)
                    }
                    epoch.put("measurements", measurements)
                    
                    val statusArr = JSONArray()
                    while (!gnssStatusBuffer.isEmpty()) statusArr.put(gnssStatusBuffer.poll())
                    epoch.put("gnssStatus", statusArr)
                    
                    val navArr = JSONArray()
                    while (!navMessageBuffer.isEmpty()) navArr.put(navMessageBuffer.poll())
                    epoch.put("navMessages", navArr)
                    
                    val imuArr = JSONArray()
                    while (!imuBuffer.isEmpty()) imuArr.put(imuBuffer.poll())
                    epoch.put("imu", imuArr)
                    
                    epochBuffer.add(epoch)
                    sendEventToRN("gnss_data", "Sats: $satCount | Status: ${statusArr.length()} | Nav: ${navArr.length()}")
                    
                    if (epochBuffer.size > 50) flush()
                }
            }
        }

        locManager.registerGnssMeasurementsCallback(gnssMeasurementsCallback!!, Handler(Looper.getMainLooper()))

        gnssStatusCallback = object : GnssStatus.Callback() {
            override fun onSatelliteStatusChanged(status: GnssStatus) {
                bufferLock.withLock {
                    val statusObj = JSONObject()
                    statusObj.put("timestamp", System.currentTimeMillis())
                    val satellites = JSONArray()
                    
                    for (i in 0 until status.satelliteCount) {
                        val sat = JSONObject().apply {
                            put("svid", status.getSvid(i))
                            put("constellationType", status.getConstellationType(i))
                            put("cn0DbHz", status.getCn0DbHz(i))
                            put("elevationDegrees", status.getElevationDegrees(i))
                            put("azimuthDegrees", status.getAzimuthDegrees(i))
                            put("hasEphemerisData", status.hasEphemerisData(i))
                            put("hasAlmanacData", status.hasAlmanacData(i))
                            put("usedInFix", status.usedInFix(i))
                            
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                                put("carrierFrequencyHz", if (status.hasCarrierFrequencyHz(i)) status.getCarrierFrequencyHz(i) else null)
                            }
                            
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                                put("basebandCn0DbHz", if (status.hasBasebandCn0DbHz(i)) status.getBasebandCn0DbHz(i) else null)
                            }
                        }
                        satellites.put(sat)
                    }
                    
                    statusObj.put("satellites", satellites)
                    gnssStatusBuffer.add(statusObj)
                }
            }
        }
        
        locManager.registerGnssStatusCallback(gnssStatusCallback!!, Handler(Looper.getMainLooper()))

        navMessageCallback = object : GnssNavigationMessage.Callback() {
            override fun onGnssNavigationMessageReceived(message: GnssNavigationMessage) {
                bufferLock.withLock {
                    val navMsg = JSONObject().apply {
                        put("svid", message.svid)
                        put("type", message.type)
                        put("status", message.status)
                        put("messageId", message.messageId)
                        put("submessageId", message.submessageId)
                        put("data", Base64.encodeToString(message.data, Base64.NO_WRAP))
                        put("timestamp", System.currentTimeMillis())
                    }
                    navMessageBuffer.add(navMsg)
                }
            }
            
            override fun onStatusChanged(status: Int) {
                Log.d(TAG, "Nav message status: $status")
            }
        }
        
        locManager.registerGnssNavigationMessageCallback(navMessageCallback!!, Handler(Looper.getMainLooper()))

        val acc = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyr = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        sensorManager.registerListener(this, acc, SensorManager.SENSOR_DELAY_GAME)
        sensorManager.registerListener(this, gyr, SensorManager.SENSOR_DELAY_GAME)
    }

    private fun stopLogging() {
        if (!running) return
        running = false
        
        bufferLock.withLock {
            flush()
        }
        
        sensorManager.unregisterListener(this)
        
        gnssMeasurementsCallback?.let {
            locManager.unregisterGnssMeasurementsCallback(it)
        }
        
        gnssStatusCallback?.let {
            locManager.unregisterGnssStatusCallback(it)
        }
        
        navMessageCallback?.let {
            locManager.unregisterGnssNavigationMessageCallback(it)
        }
        
        sendEventToRN("gnss_log", "GNSS logging stopped. File: ${outFile?.absolutePath}")
        stopForeground(true)
        stopSelf()
    }

    private fun flush() {
        val arr = JSONArray()
        while (!epochBuffer.isEmpty()) arr.put(epochBuffer.poll())
        outFile?.let {
            val fos = FileOutputStream(it, true)
            fos.write((arr.toString() + "\n").toByteArray())
            fos.close()
        }
    }

    private fun sendEventToRN(eventName: String, message: String) {
        reactContext?.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            ?.emit(eventName, message)
    }

    override fun onSensorChanged(e: SensorEvent?) {
        e ?: return
        val j = JSONObject()
        j.put("t", System.nanoTime())
        j.put("type", e.sensor.type)
        val vals = JSONArray()
        for (v in e.values) vals.put(v)
        j.put("v", vals)
        imuBuffer.add(j)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
