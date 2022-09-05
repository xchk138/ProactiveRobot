package ph.edu.dlsu.robot

import android.Manifest
import android.app.Activity
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothSocket
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.Toast
import androidx.core.app.ActivityCompat
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import java.io.File
import java.io.IOException
import java.io.OutputStream
import java.util.*
import java.util.concurrent.TimeUnit


class MainActivity : Activity(), CameraBridgeViewBase.CvCameraViewListener2 {
    private var mOpenCvCameraView: CameraBridgeViewBase? = null
    private var mBlePermmited = false
    private fun PackageManager.missingSystemFeature(name: String): Boolean = !hasSystemFeature(name)
    private var mBleAdapter: BluetoothAdapter? = null
    private var mBleManager: BluetoothManager? = null
    private var mPermissions: Array<String> ?= null
    private var mNextMove: Int = MOVE_NONE
    private var mRobotBleDevice: BluetoothDevice ?= null
    private var mRobotBleConn: BleConnThread ?= null
    private var mRobotControlThread: RobotControlThread ?= null
    private var mJobDone :Boolean = false
    private var mTracker :Long = 0
    private var absPathYolo :String ?= null
    private var absPathSiamMain: String ?= null
    private var absPathSiamCls: String ?= null
    private var absPathSiamReg: String ?= null

    private fun getAssetsFile(path: String): ByteArray {
        return try {
            Log.i(TAG, "reading assets file: $path")
            val inFile = assets.open(path)
            Log.i(TAG, "file opened success: $path")
            val data = inFile.readBytes()
            Log.i(TAG, "file read success: $path")
            inFile.close()
            data
        } catch (e : IOException) {
            Log.e(TAG, "failed to open path in assets: $path")
            ByteArray(0)
        }
    }

    private fun readAssetsFile(path: String,
                               data: ByteArray,
                               offset: Int,
                               size: Int) : Boolean{
        return try {
            Log.i(TAG, "reading assets file: $path")
            val inFile = assets.open(path)
            Log.i(TAG, "file opened success: $path")
            inFile.read(data, offset, size)
            Log.i(TAG, "file read success: $path")
            inFile.close()
            true
        } catch (e : IOException) {
            Log.e(TAG, "failed to open path in assets: $path")
            false
        }
    }

    private fun sizeAssetsFile(path: String): Int {
        return try {
            Log.i(TAG, "reading assets file: $path")
            val inFile = assets.open(path)
            Log.i(TAG, "file opened success: $path")
            val size = inFile.available()
            Log.i(TAG, "file read success: $path")
            inFile.close()
            size
        } catch (e : IOException) {
            Log.e(TAG, "failed to open path in assets: $path")
            0
        }
    }

    // transfer the files from assets into /data/...
    // return the target file path
    private fun transFile(
        relPath :String,
        newName :String
    ) :String{
// for YOLO
        val data = getAssetsFile(relPath)
        val fileModel = File(cacheDir, newName)
        val absPath = fileModel.absolutePath

        if (!fileModel.exists()) {
            try {
                if (fileModel.createNewFile())
                    Log.i(TAG, "file created success: $absPath")
                else
                    Log.e(TAG, "failed to create file: $absPath")
            } catch (e: IOException) {
                Log.e(TAG, "file creation error: ${e.message}")
            }
        }else{
            Log.i(TAG, "file already exists: $absPath")
        }

        if(fileModel.canWrite()) {
            val ostream = fileModel.outputStream()
            ostream.write(data)
            ostream.flush()
            ostream.close()
            Log.i(TAG, "model saved success to $absPath")
        } else {
            Log.e(TAG, "cannot write into file: $absPath")
            return ""
        }

        return absPath
    }

    private fun loadModelsFromFiles(){
        Log.i(TAG,"loading yolo tracker...")

        val pathYolo = model_base + yolov5_split + yolov5_main
        val pathSiamMain = model_base + siam_split + siam_main
        val pathSiamCls = model_base + siam_split + siam_cls
        val pathSiamReg = model_base + siam_split + siam_reg

        absPathYolo = transFile(pathYolo, "yolo5n-face-256.onnx")
        absPathSiamMain = transFile(pathSiamMain, "siam_main.onnx")
        absPathSiamCls = transFile(pathSiamCls, "siam_cls.onnx")
        absPathSiamReg = transFile(pathSiamReg, "siam_reg.onnx")
    }

    private val mLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("native-lib")
                    // Front Camera
                    mOpenCvCameraView!!.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT)
                    mOpenCvCameraView!!.enableView()
                    // full APIs can be used till now
                    // loading trackers
                    loadModelsFromFiles()
                    mTracker = jniGetYoloTracker(
                        assets,
                        absPathYolo!!,
                        absPathSiamMain!!,
                        absPathSiamCls!!,
                        absPathSiamReg!!)
                    Log.i(TAG,"creating yolo tracker success!")
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    private fun checkBleSupport(){
        // Check to see if the Bluetooth classic feature is available.
        packageManager.takeIf { it.missingSystemFeature(PackageManager.FEATURE_BLUETOOTH) }?.also {
            Toast.makeText(this, R.string.bluetooth_not_supported, Toast.LENGTH_LONG).show()
            finish()
        }
        // Check to see if the BLE feature is available.
        packageManager.takeIf { it.missingSystemFeature(PackageManager.FEATURE_BLUETOOTH_LE) }?.also {
            Toast.makeText(this, R.string.ble_not_supported, Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private fun getBleMgr(){
        mBleManager = getSystemService(BluetoothManager::class.java)
        mBleAdapter = mBleManager!!.adapter
        if (mBleAdapter == null) {
            // Device doesn't support Bluetooth
            Toast.makeText(this, R.string.bluetooth_not_supported, Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    private fun checkPermissions(){
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            mPermissions = arrayOf(
                Manifest.permission.CAMERA
                ,Manifest.permission.BLUETOOTH_SCAN
                ,Manifest.permission.BLUETOOTH_CONNECT
                ,Manifest.permission.BLUETOOTH_ADVERTISE
            )
        } else {
            mPermissions = arrayOf(
                Manifest.permission.CAMERA
                ,Manifest.permission.ACCESS_COARSE_LOCATION
                ,Manifest.permission.BLUETOOTH
                ,Manifest.permission.BLUETOOTH_ADMIN
            )
        }
        ActivityCompat.requestPermissions(
            this@MainActivity,
            mPermissions!!,
            CAM_BLE_PERM_REQ
        )
    }

    private fun cmdForward(outStream : OutputStream){
        outStream.write("ONA".toByteArray())
        outStream.flush()
        Log.i(TAG, "Forward")
    }

    private fun cmdStop(outStream : OutputStream){
        try {
            outStream.write("ONF".toByteArray())
        } catch (e:IOException){
            Log.e(TAG, "failed to send data to target BLE device")
        }
        outStream.flush()
        Log.i(TAG, "Stop")
    }

    private fun cmdStepForward(outStream : OutputStream){
        Log.i(TAG, "Step Forward Begin")
        cmdForward(outStream)
        Log.i(TAG, "sleep begin")
        TimeUnit.MILLISECONDS.sleep(ACTION_HOLD)
        Log.i(TAG, "sleep end")
        cmdStop(outStream)
        Log.i(TAG, "Step Forward End")
    }

    private fun cmdBackward(outStream: OutputStream){
        outStream.write("ONB".toByteArray())
        outStream.flush()
        Log.i(TAG, "Backward")
    }

    private fun cmdStepBackward(outStream : OutputStream){
        Log.i(TAG, "Step Back Begin")
        cmdBackward(outStream)
        TimeUnit.MILLISECONDS.sleep(ACTION_HOLD)
        cmdStop(outStream)
        Log.i(TAG, "Step Back End")

    }

    private fun cmdLeft(outStream: OutputStream){
        outStream.write("ONC".toByteArray())
        outStream.flush()
        Log.i(TAG, "Left")
    }

    private fun cmdRight(outStream: OutputStream){
        outStream.write("OND".toByteArray())
        outStream.flush()
        Log.i(TAG, "Right")
    }

    private fun cmdStepLeft(outStream: OutputStream){
        Log.i(TAG, "Step Left Begin")
        cmdLeft(outStream)
        TimeUnit.MILLISECONDS.sleep(ACTION_HOLD/3)
        cmdStop(outStream)
        //cmdStepForward(outStream)
        //cmdRight(outStream)
        //TimeUnit.MILLISECONDS.sleep(100);
        //cmdStop(outStream)
        Log.i(TAG, "Step Left End")
    }

    private fun cmdStepRight(outStream: OutputStream){
        Log.i(TAG, "Step Right Begin")
        cmdRight(outStream)
        TimeUnit.MILLISECONDS.sleep(ACTION_HOLD/3)
        cmdStop(outStream)
        //cmdStepForward(outStream)
        //cmdLeft(outStream)
        //TimeUnit.MILLISECONDS.sleep(100);
        //cmdStop(outStream)
        Log.i(TAG, "Step Right End")
    }

    private fun cmdSelfCheck(outStream: OutputStream){
        Log.i(TAG, "Robot Self Check Begin")
        cmdStepForward(outStream)
        cmdStepBackward(outStream)
        cmdStepLeft(outStream)
        cmdStepRight(outStream)
        Log.i(TAG, "Robot Self Check End")
    }

    private inner class RobotControlThread(outStream: OutputStream) : Thread() {
        private val oStream:OutputStream = outStream
        override fun run() {
        // send command data to robot
            Log.i(TAG, "Robot Control Thread Running...")
            //cmdSelfCheck(oStream)
            while(!mJobDone){
                when(mNextMove){
                    MOVE_NONE -> {
                        cmdStop(oStream)
                        //TimeUnit.MILLISECONDS.sleep(ACTION_HOLD)
                    }
                    MOVE_FORWARD -> {
                        cmdStepForward(oStream)
                    }
                    MOVE_BACK -> {
                        cmdStepBackward(oStream)
                    }
                    MOVE_LEFT -> {
                        cmdStepLeft(oStream)
                    }
                    MOVE_RIGHT -> {
                        cmdStepRight(oStream)
                    }
                }
                TimeUnit.MILLISECONDS.sleep(35);
            }
            Log.i(TAG, "Robot Control Thread Finished.")
        }
    }

    private inner class BleConnThread(device: BluetoothDevice) : Thread() {
        private var uuid :UUID ?= UUID.fromString(BLE_UUID)
        private val mmSocket: BluetoothSocket? by lazy(LazyThreadSafetyMode.NONE) {
            device.createRfcommSocketToServiceRecord(uuid)
        }

        override fun run() {
            Log.i(TAG, "socket created with UUID: $uuid")
            // Cancel discovery because it otherwise slows down the connection.
            mBleAdapter!!.cancelDiscovery()
            Log.i(TAG, "Ble scanning canceled!")
            mmSocket?.let { socket ->
                // Connect to the remote device through the socket. This call blocks
                // until it succeeds or throws an exception.
                Log.i(TAG, "BLE socket connecting!")
                try {
                    socket.connect()
                } catch (e:Exception){
                    Log.e(TAG, "bluetooth connection failed with ${e.message}")
                }
                // The connection attempt succeeded. Perform work associated with
                // the connection in a separate thread.
                if(socket.isConnected){
                    Log.w(TAG, "target BLE device connected!!!")
                    val outStream = socket.outputStream
                    // start a new thread to serve robot controller
                    mRobotControlThread = RobotControlThread(socket.outputStream)
                    mRobotControlThread!!.start()
                    //controlThread.join()
                    Log.i(TAG, "connection finished!")
                } else {
                    Log.w(TAG, "target BLE device failed to connect!!!")
                }
            }
        }

        // Close the client socket and causes the thread to finish.
        fun cancel() {
            try {
                mmSocket?.close()
            } catch (e: IOException) {
                Log.e(TAG, "Could not close the client socket", e)
            }
        }
    }

    private fun pairDevice(device :BluetoothDevice) {
        try {
            Log.d(TAG, "Start Pairing... with: " + device.name)
            device.createBond()
            Log.d(TAG, "Pairing finished.")
        } catch (e: Exception) {
            Log.e(TAG, e.localizedMessage!!)
        }
    }

    private val mPairingRequestReceiver: BroadcastReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val action = intent.action
            if (action == BluetoothDevice.ACTION_PAIRING_REQUEST) {
                try {
                    mRobotBleDevice = intent.getParcelableExtra<BluetoothDevice>(BluetoothDevice.EXTRA_DEVICE)
                    val pinCode :ByteArray = BLE_PIN.toByteArray()
                    mRobotBleDevice!!.setPin(pinCode)
                    mRobotBleDevice!!.setPairingConfirmation(true) // this requires BLUETOOTH_PRIVILEGE
                    // connect to this ble device
                    mRobotBleConn = BleConnThread(mRobotBleDevice!!)
                    mRobotBleConn!!.start()
                    //connThread.join()
             } catch (e: Exception) {
                    Log.e(TAG, "Error occurs when trying to auto pair: ${e.message}")
                    e.printStackTrace()
                }
            }
        }
    }


    // Create a BroadcastReceiver for ACTION_FOUND.
    private val mBleScanReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val action: String ?= intent.action
            Log.i(TAG, "action = ${intent.action}")
            when(action) {
                BluetoothDevice.ACTION_FOUND -> {
                    // Discovery has found a device. Get the BluetoothDevice
                    // object and its info from the Intent.
                    val device: BluetoothDevice? =
                        intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE)
                    val deviceName = device?.name
                    val deviceAddr = device?.address // MAC address
                    Log.i(TAG, "scan BLE device $deviceName : $deviceAddr")
                    // check if desired device is found
                    if (deviceName == BLE_TARGET_DEVICE){
                        Log.w(TAG, "Desired BLE device found: $deviceName - $deviceAddr !!!")
                        // register pairing callback to connect to device with PIN code
                        val filter = IntentFilter(BluetoothDevice.ACTION_PAIRING_REQUEST)
                        registerReceiver(mPairingRequestReceiver, filter)
                        // begin to pair device
                        pairDevice(device)
                    }
                }
            }
        }
    }

    private fun turnOnBle(){
        if (mBleAdapter?.isEnabled == false) {
            val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
            startActivityForResult(enableBtIntent, REQUEST_ENABLE_BLE)
        } else {
            scanBleDevices()
        }
    }

    private fun scanBleDevices(){
        Log.i(TAG, "scanning devices...")
        // get bound devices
        val pairedDevices: Set<BluetoothDevice>? = mBleAdapter?.bondedDevices
        pairedDevices?.forEach { device ->
            val deviceName = device.name
            val deviceAddr = device.address // MAC address
            Log.i(TAG, "bound BLE device $deviceName : $deviceAddr")
            // check if desired device is found
            if (deviceName == BLE_TARGET_DEVICE){
                mRobotBleDevice = device
                // connect to this ble device
                mRobotBleConn = BleConnThread(mRobotBleDevice!!)
                mRobotBleConn!!.start()
                //mRobotBleConn!!.join()
                // abort the discovery
                return
            }
        }
        // scan unbound devices
        // Register for broadcasts when a device is discovered.
        val filter = IntentFilter(BluetoothDevice.ACTION_FOUND)
        registerReceiver(mBleScanReceiver, filter)
        // then start to scan devices
        if(!mBleAdapter!!.startDiscovery()){
            Log.e(TAG, "Failed to invoke Bluetooth Discovery Process!")
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        Log.i(TAG, "called onCreate")
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        // this is hardware support check, goes always the first
        checkBleSupport()
        // check if bluetooth is supported on software layer
        getBleMgr()

        mOpenCvCameraView = findViewById<CameraBridgeViewBase>(R.id.main_surface)
        mOpenCvCameraView!!.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView!!.setCvCameraViewListener(this)
        mOpenCvCameraView!!.setMaxFrameSize(800, 600);

        // request permission for use of camera device
        checkPermissions()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        when (requestCode){
            REQUEST_ENABLE_BLE -> {
                if (resultCode == RESULT_OK){
                    scanBleDevices()
                } else {
                    Toast.makeText(this, R.string.cannot_access_bluetooth, Toast.LENGTH_LONG)
                }
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        var blePermCnt = 0
        var blePermCntMin = mPermissions!!.size - 1
        Log.i(TAG, "mPermissions size = ${mPermissions!!.size}")
        when (requestCode) {
            CAM_BLE_PERM_REQ -> {
                if (grantResults.isEmpty()) {
                    Log.e(TAG, "granted results empty!!!")
                    return
                }
                val grantSize:Int = grantResults.size
                Log.i(TAG, "size of grantResults: $grantSize")
                // check camera permission
                var res_id = 0
                if (grantResults[res_id] == PackageManager.PERMISSION_GRANTED) {
                    Log.i(TAG, "Camera access OK!")
                    mOpenCvCameraView!!.setCameraPermissionGranted()
                } else {
                    val message = "Camera permission was not granted"
                    Log.e(TAG, message)
                    Toast.makeText(this, message, Toast.LENGTH_LONG).show()
                }
                res_id ++

                // check bluetooth permission
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    if(res_id >= grantSize) return
                    if(grantResults[res_id] == PackageManager.PERMISSION_GRANTED){
                        Log.i(TAG, "BLUETOOTH_SCAN access OK!")
                        blePermCnt ++
                    } else {
                        val message = "BLUETOOTH_SCAN permission was not granted"
                        Log.e(TAG, message)
                        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
                    }
                    res_id ++

                    if(res_id >= grantSize) return
                    if(grantResults[res_id] == PackageManager.PERMISSION_GRANTED){
                        Log.i(TAG, "BLUETOOTH_CONNECT access OK!")
                        blePermCnt ++
                    } else {
                        val message = "BLUETOOTH_CONNECT permission was not granted"
                        Log.e(TAG, message)
                        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
                    }
                    res_id ++

                    if(res_id >= grantSize) return
                    if(grantResults[res_id] == PackageManager.PERMISSION_GRANTED){
                        Log.i(TAG, "BLUETOOTH_ADVERTISE access OK!")
                        blePermCnt ++
                    } else {
                        val message = "BLUETOOTH_ADVERTISE permission was not granted"
                        Log.e(TAG, message)
                        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
                    }
                    res_id ++
                } else {
                    if(res_id >= grantSize) return
                    if(grantResults[res_id] == PackageManager.PERMISSION_GRANTED){
                        Log.i(TAG, "ACCESS_COARSE_LOCATION access OK!")
                        blePermCnt ++
                    } else {
                        val message = "ACCESS_COARSE_LOCATION permission was not granted"
                        Log.e(TAG, message)
                        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
                    }
                    res_id ++

                    if(res_id >= grantSize) return
                    if(grantResults[res_id] == PackageManager.PERMISSION_GRANTED){
                        Log.i(TAG, "BLUETOOTH access OK!")
                        blePermCnt ++
                    } else {
                        val message = "BLUETOOTH permission was not granted"
                        Log.e(TAG, message)
                        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
                    }
                    res_id ++

                    if(res_id >= grantSize) return
                    if(grantResults[res_id] == PackageManager.PERMISSION_GRANTED){
                        Log.i(TAG, "BLUETOOTH_ADMIN access OK!")
                        blePermCnt ++
                    } else {
                        val message = "BLUETOOTH_ADMIN permission was not granted"
                        Log.e(TAG, message)
                        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
                    }
                    res_id ++
                }
            }
            else -> {
                Log.e(TAG, "Unexpected permission request")
            }
        }
        Log.i(TAG, "blePermCnt size = $blePermCnt")
        if (blePermCnt >= blePermCntMin ) {
            mBlePermmited = true
            turnOnBle()
        }
    }

    override fun onPause() {
        super.onPause()
        if (mOpenCvCameraView != null)
            mOpenCvCameraView!!.disableView()
    }

    override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (mOpenCvCameraView != null)
            mOpenCvCameraView!!.disableView()
        // Don't forget to unregister the ACTION_FOUND receiver.
        unregisterReceiver(mBleScanReceiver)
        mJobDone = true
        // release tracker
        jniFreeYoloTracker(mTracker)
    }

    override fun onCameraViewStarted(width: Int, height: Int) {}

    override fun onCameraViewStopped() {}

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        val img = inputFrame.rgba()
        // tracking all objects
        mNextMove = jniTrackAll(img.nativeObjAddr, mTracker)

        return img
    }

    private external fun jniGetYoloTracker(
        assets: AssetManager,
        pathYolo : String,
        pathSiamMain : String,
        pathSiamCls : String,
        pathSiamReg : String) : Long
    private external fun jniFreeYoloTracker(tracker: Long)
    private external fun jniTrackAll(frame: Long, tracker: Long) : Int

    companion object {
        private const val TAG = "CamApp"
        private const val CAM_BLE_PERM_REQ = 1
        private const val REQUEST_ENABLE_BLE = 2
        private const val BLE_UUID = "00001101-0000-1000-8000-00805F9B34FB"
        private const val BLE_PIN = "1234"
        private const val BLE_TARGET_DEVICE = "BT04-A"
        private const val ACTION_HOLD : Long = 15

        private const val MOVE_NONE = 0
        private const val MOVE_FORWARD = 1
        private const val MOVE_BACK = 2
        private const val MOVE_LEFT = 3
        private const val MOVE_RIGHT = 4

        // global model loading configuration
        private const val model_base = "models/"
        // DaSiamRPN
        private const val siam_split = "SiamRPN/"
        private const val siam_main = "dasiamrpn_model.onnx"
        private const val siam_cls = "dasiamrpn_kernel_cls1.onnx"
        private const val siam_reg = "dasiamrpn_kernel_r1.onnx"
        // YOLOv5-6.1
        private const val yolov5_split = "YOLOv5/"
        private const val yolov5_main = "yolov5n-face-256.onnx"
    }
}