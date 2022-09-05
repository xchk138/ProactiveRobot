# CameraPreview

Android NDK OpenCV Camera Preview


![](https://github.com/melvincabatuan/CameraPreview/blob/master/device-2019-10-02-080042.png)

## Requirements

- Android Studio with NDK, CMake and LLDB. [Link](https://developer.android.com/studio/projects/install-ndk.md)
- OpenCV - [OpenCV 4.1.1 Android release](https://sourceforge.net/projects/opencvlibrary/files/4.1.1/opencv-4.1.1-android-sdk.zip/download)
 or [later](https://opencv.org/releases) 
 
 ```
 2019-10-02 07:55:45.877  General configuration for OpenCV 4.1.1 =====================================
2019-10-02 07:55:45.877    Version control:               4.1.1
2019-10-02 07:55:45.877    Platform:
2019-10-02 07:55:45.877      Timestamp:                   2019-07-26T03:48:10Z
2019-10-02 07:55:45.877      Host:                        Linux 4.15.0-54-generic x86_64
2019-10-02 07:55:45.877      Target:                      Android 1 aarch64
2019-10-02 07:55:45.877      CMake:                       3.6.0-rc2
2019-10-02 07:55:45.877      CMake generator:             Ninja
2019-10-02 07:55:45.878      CMake build tool:            /opt/android/android-sdk.gradle/cmake/3.6.4111459/bin/ninja
2019-10-02 07:55:45.878      Configuration:               Release
2019-10-02 07:55:45.878    CPU/HW features:
2019-10-02 07:55:45.878      Baseline:                    NEON FP16
2019-10-02 07:55:45.878    C/C++:
2019-10-02 07:55:45.878      Built as dynamic libs?:      NO
2019-10-02 07:55:45.878      C++ Compiler:                /opt/android/android-ndk-r18b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++  (ver 7.0)
2019-10-02 07:55:45.878      C++ flags (Release):         -isystem /opt/android/android-ndk-r18b/sysroot/usr/include/aarch64-linux-android -DANDROID -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -Wa,--noexecstack -Wformat -Werror=format-security -std=c++11    -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self -Winconsistent-missing-override -Wno-delete-non-virtual-dtor -Wno-unnamed-type-template-args -Wno-comment -fdiagnostics-show-option -Qunused-arguments    -fvisibility=hidden -fvisibility-inlines-hidden  -O2 -DNDEBUG   -DNDEBUG
2019-10-02 07:55:45.878      C++ flags (Debug):           -isystem /opt/android/android-ndk-r18b/sysroot/usr/include/aarch64-linux-android -DANDROID -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -Wa,--noexecstack -Wformat -Werror=format-security -std=c++11    -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self -Winconsistent-missing-override -Wno-delete-non-virtual-dtor -Wno-unnamed-type-template-args -Wno-comment -fdiagnostics-show-option -Qunused-arguments    -fvisibility=hidden -fvisibility-inlines-hidden  -O0 -fno-limit-debug-info   -DDEBUG -D_DEBUG -g
2019-10-02 07:55:45.878      C Compiler:                  /opt/android/android-ndk-r18b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang
2019-10-02 07:55:45.878      C flags (Release):           -isystem /opt/android/android-ndk-r18b/sysroot/usr/include/aarch64-linux-android -DANDROID -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -Wa,--noexecstack -Wformat -Werror=format-security    -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self -Winconsistent-missing-override -Wno-delete-non-virtual-dtor -Wno-unnamed-type-template-args -Wno-comment -fdiagnostics-show-option -Qunused-arguments    -fvisibility=hidden -fvisibility-inlines-hidden  -O2 -DNDEBUG   -DNDEBUG
2019-10-02 07:55:45.878      C flags (Debug):             -isystem /opt/android/android-ndk-r18b/sysroot/usr/include/aarch64-linux-android -DANDROID -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -Wa,--noexecstack -Wformat -Werror=format-security    -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self -Winconsistent-missing-override -Wno-delete-non-virtual-dtor -Wno-unnamed-type-template-args -Wno-comment -fdiagnostics-show-option -Qunused-arguments    -fvisibility=hidden -fvisibility-inlines-hidden  -O0 -fno-limit-debug-info   -DDEBUG -D_DEBUG -g
2019-10-02 07:55:45.878      Linker flags (Release):      -Wl,--exclude-libs,libgcc.a -Wl,--exclude-libs,libatomic.a -nostdlib++ --sysroot /opt/android/android-ndk-r18b/platforms/android-21/arch-arm64 -Wl,--build-id -Wl,--warn-shared-textrel -Wl,--fatal-warnings -L/opt/android/android-ndk-r18b/sources/cxx-stl/llvm-libc++/libs/arm64-v8a -Wl,--no-undefined -Wl,-z,noexecstack -Qunused-arguments -Wl,-z,relro -Wl,-z,now    
2019-10-02 07:55:45.878      Linker flags (Debug):        -Wl,--exclude-libs,libgcc.a -Wl,--exclude-libs,libatomic.a -nostdlib++ --sysroot /opt/android/android-ndk-r18b/platforms/android-21/arch-arm64 -Wl,--build-id -Wl,--warn-shared-textrel -Wl,--fatal-warnings -L/opt/android/android-ndk-r18b/sources/cxx-stl/llvm-libc++/libs/arm64-v8a -Wl,--no-undefined -Wl,-z,noexecstack -Qunused-arguments -Wl,-z,relro -Wl,-z,now    
2019-10-02 07:55:45.878      ccache:                      YES
2019-10-02 07:55:45.878      Precompiled headers:         NO
2019-10-02 07:55:45.878      Extra dependencies:          z dl m log
2019-10-02 07:55:45.878      3rdparty dependencies:       tbb libcpufeatures ittnotify libprotobuf libjpeg-turbo libwebp libpng libtiff libjasper IlmImf quirc tegra_hal
2019-10-02 07:55:45.878    OpenCV modules:
2019-10-02 07:55:45.878      To be built:                 calib3d core dnn features2d flann highgui imgcodecs imgproc java ml objdetect photo stitching video videoio
2019-10-02 07:55:45.878      Disabled:                    world
2019-10-02 07:55:45.878      Disabled by dependency:      -
2019-10-02 07:55:45.878      Unavailable:                 gapi js python2 python3 ts
2019-10-02 07:55:45.878      Applications:                -
2019-10-02 07:55:45.878      Documentation:               NO
2019-10-02 07:55:45.878      Non-free algorithms:         NO
2019-10-02 07:55:45.878    Android NDK:                   /opt/android/android-ndk-r18b (ver 18.1.5063045)
2019-10-02 07:55:45.878      Android ABI:                 arm64-v8a
2019-10-02 07:55:45.878      NDK toolchain:               aarch64-linux-android-clang
2019-10-02 07:55:45.878      STL type:                    c++_shared
2019-10-02 07:55:45.878      Native API level:            21
2019-10-02 07:55:45.878    Android SDK:                   /opt/android/android-sdk.gradle (tools: 26.1.1 build tools: 28.0.3)
2019-10-02 07:55:45.878    GUI: 
2019-10-02 07:55:45.878    Media I/O: 
2019-10-02 07:55:45.878      ZLib:                        z (ver 1.2.7)
2019-10-02 07:55:45.878      JPEG:                        build-libjpeg-turbo (ver 2.0.2-62)
2019-10-02 07:55:45.878      WEBP:                        build (ver encoder: 0x020e)
2019-10-02 07:55:45.878      PNG:                         build (ver 1.6.37)
2019-10-02 07:55:45.878      TIFF:                        build (ver 42 - 4.0.10)
2019-10-02 07:55:45.878      JPEG 2000:                   build (ver 1.900.1)
2019-10-02 07:55:45.879      OpenEXR:                     build (ver 2.3.0)
2019-10-02 07:55:45.879      HDR:                         YES
2019-10-02 07:55:45.879      SUNRASTER:                   YES
2019-10-02 07:55:45.879      PXM:                         YES
2019-10-02 07:55:45.879      PFM:                         YES
2019-10-02 07:55:45.879    Video I/O:
2019-10-02 07:55:45.879    Parallel framework:            TBB (ver 2019.0 interface 11008)
2019-10-02 07:55:45.879    Trace:                         YES (with Intel ITT)
2019-10-02 07:55:45.879    Other third-party libraries:
2019-10-02 07:55:45.879      Custom HAL:                  YES (carotene (ver 0.0.1))
2019-10-02 07:55:45.879      Protobuf:                    build (3.5.1)
2019-10-02 07:55:45.879    Python (for build):            /usr/bin/python2.7
2019-10-02 07:55:45.879    Java:                          export all functions
2019-10-02 07:55:45.879      ant:                         NO
2019-10-02 07:55:45.879      Java wrappers:               YES
2019-10-02 07:55:45.879      Java tests:                  NO
2019-10-02 07:55:45.879    Install to:                    /build/master_pack-android/build/o4a/install
2019-10-02 07:55:45.879  -----------------------------------------------------------------
2019-10-02 07:55:45.879 30158-30158/ph.edu.dlsu.helloopencv D/MainActivity: OpenCV library found inside package. Using it!
2019-10-02 07:55:45.879 30158-30158/ph.edu.dlsu.helloopencv I/MainActivity: OpenCV loaded successfully
 ```
