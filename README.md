# Java Launcher for Android

A universal Java application launcher for Android devices. Run Java JAR files directly on your Android phone or tablet without requiring Termux or manual Java configuration.

## Features

- **Universal JAR Support**: Import and run any Java application compatible with OpenJDK 17
- **No Bundled Apps**: You bring your own JAR files - no pre-installed applications
- **Built-in OpenJDK 17**: Automatically downloads and configures OpenJDK for ARM64
- **Headless Mode**: Run command-line Java applications with full stdout/stderr capture
- **GUI Mode**: Launch Swing/AWT applications via external X11 display server (Termux:X11)
- **File Import**: Import JAR files from your device storage using Android's Storage Access Framework
- **App Management**: Install, uninstall, and manage multiple Java applications
- **Log Viewer**: Real-time process output with copy/share support
- **Process Control**: Start/stop Java processes with full lifecycle management

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Java Launcher App                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │  Jetpack    │  │   File      │  │   Process       │   │
│  │  Compose UI │  │   Browser   │  │   Control       │   │
│  └────┬──────┘  └────┬──────┘  └─────────────────┘   │
│       └─────────────┴──────────────────────────────────┘   │
│                    ViewModel Layer                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │ HomeVM  │ │ FileVM  │ │ LogVM   │ │ Settings│          │
│  │         │ │         │ │         │ │   VM    │          │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘          │
└───────┼─────────────┼──────────┼─────────────┼──────────────┘
        │             │          │             │
        └─────────────┴──────────┴─────────────┘
                    Engine Layer
  ┌──────────────────────────────────────────────────────┐
  │  JavaAppManager  │  DownloadManager  │  JarExecutor  │
  │  - JDK lifecycle │  - Resume support │  - Process    │
  │  - App catalog   │  - Progress flow  │    control    │
  └──────────────────────────────────────────────────────┘
        │
  ┌─────┴──────────────────────────────────────────────────┐
  │              Native Layer (JNI) - Optional            │
  │  ┌──────────────┐  ┌─────────────────────────────┐   │
  │  │ Swing Bridge │  │      VNC Client (RFB)       │   │
  │  │ (Future)     │  │  ANativeWindow + GPU tex    │   │
  │  └──────────────┘  └─────────────────────────────┘   │
  └──────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Android Studio Hedgehog (2023.1.1) or later
- Android SDK API 34
- NDK 25.2.9519653 (for JNI components, optional)
- JDK 17 (for building)

### Building

1. Clone the repository:
```bash
git clone https://github.com/yourusername/java-launcher-android.git
cd java-launcher-android
```

2. Open in Android Studio and sync Gradle

3. Build the APK:
```bash
./gradlew assembleDebug
```

### Running

1. Install the APK on your Android device (ARM64 required)
2. Launch the app
3. Go to **Setup** and install OpenJDK (requires ~200MB download)
4. Go to **Files** tab and tap **+** to import a JAR file from your device
5. Select an app on the **Home** screen and tap **Start**

## Importing JAR Files

### From Device Storage

1. Go to the **Files** tab
2. Tap the **+** (FAB) button
3. Select a `.jar` file from your device
4. The app will be automatically installed and available on the Home screen

### Supported Applications

| Application Type | Compatibility | Notes |
|-----------------|---------------|-------|
| Pure Java CLI | Full | Best compatibility |
| Swing/AWT GUI | Partial | Requires external X11 server |
| JavaFX | Limited | May require additional modules |
| Native JNI | None | Android cannot run desktop native libs |

### Tested Applications

Users have reported success with:
- Minecraft Server (headless mode)
- Various command-line tools
- Simple Swing applications (with X11)

## GUI Display Modes

### 1. Headless Mode (Default)

Runs Java applications without a display server:
```bash
java -Djava.awt.headless=true -jar app.jar
```

**Best for**: Command-line tools, servers, batch processing, any app that works without GUI

### 2. GUI Mode via Termux:X11 (Recommended)

**Architecture**:
```
Java App (Swing) → DISPLAY=:0 → Termux:X11 Server → Android Screen
```

**Setup**:
1. Install [Termux](https://f-droid.org/packages/com.termux/) from F-Droid
2. Install [Termux:X11](https://f-droid.org/packages/com.termux.x11/) from F-Droid
3. In Termux, run: `pkg install x11-repo && pkg install termux-x11-nightly`
4. Start Termux:X11 app
5. In Java Launcher, go to **GUI Mode** and launch your app

**Best for**: Full GUI applications, IDEs, anything requiring mouse/keyboard

### 3. External VNC Server

Connect to an existing VNC server:
1. Set up x11vnc or TigerVNC on a PC or via Termux
2. In Java Launcher, configure the VNC host and port
3. Launch your application

**Best for**: Remote access, headless servers with VNC

### 4. Native Bridge (Future)

Direct JNI bridge from AWT Graphics2D to Android Canvas/Skia:
- Status: Framework implemented (`swing_bridge.c`)
- Requires: Custom OpenJDK build with modified AWT peers
- Not yet functional in current release

## Configuration

### JVM Settings

- **Max Memory**: Default `512m`, adjustable up to device limits
- **Extra JVM Args**: Additional arguments passed to `java` command
- **JDK URL**: Custom OpenJDK download source

### Download Sources

Default JDK source: Termux OpenJDK bootstrap packages

Alternative sources:
- Adoptium Temurin: `https://github.com/adoptium/...`
- Custom CDN: Configure in Settings

## Project Structure

```
app/
├── src/main/
│   ├── java/com/openrocket/launcher/
│   │   ├── MainActivity.kt
│   │   ├── OpenRocketApplication.kt
│   │   ├── engine/
│   │   │   ├── DownloadManager.kt      # Resume-capable downloader
│   │   │   ├── JdkManager.kt          # JDK install/management
│   │   │   ├── JarExecutor.kt         # Process execution
│   │   │   ├── JavaAppManager.kt      # App lifecycle (import-based)
│   │   │   ├── JavaAppService.kt      # Foreground service
│   │   │   └── JavaAppSetupState.kt   # Setup state machine
│   │   ├── ui/
│   │   │   ├── navigation/            # Jetpack Navigation
│   │   │   ├── screens/               # Compose Screens
│   │   │   ├── theme/                 # Material3 Theme
│   │   │   └── viewmodel/             # MVVM ViewModels
│   │   └── vnc/
│   │       ├── VncClient.kt           # VNC RFB client
│   │       └── VncView.kt             # SurfaceView renderer
│   ├── cpp/
│   │   ├── CMakeLists.txt
│   │   ├── swing_bridge.c             # JNI AWT bridge (stub)
│   │   └── vnc_client.c               # Native VNC protocol
│   └── res/                           # Android resources
├── README.md
├── ARCHITECTURE.md
├── build.gradle.kts
└── settings.gradle.kts
```

## Troubleshooting

### "JDK not installed"
- Go to Setup tab and tap "Install JDK"
- Ensure stable WiFi connection (~200MB download)
- Check download URL in Settings if default fails

### "Cannot execute binary"
- Android 10+ requires executables in app-private directories
- JDK is extracted to `/data/data/<package>/files/jdk/`
- Ensure `java` binary has execute permissions

### "Out of memory"
- Reduce max memory in Settings (default 512m)
- Close other apps before running large Java applications

### "No display server found" (GUI mode)
- Install Termux:X11 from F-Droid
- Ensure Termux:X11 is running before launching GUI apps
- Check that the X11 Unix socket exists at `/data/data/com.termux/files/usr/tmp/.X11-unix/X0`

### "App won't start"
- Verify the JAR is a pure Java application (no native libraries)
- Check logs in the Logs tab for error messages
- Try running with `-Djava.awt.headless=true` for headless mode

## Security Considerations

### Process Isolation

- Java processes run under app UID
- No root privileges required
- SELinux policies apply to all executed binaries

### File Permissions

```
/data/data/<package>/
    ├── files/
    │   ├── jdk/          (rwxr-xr-x)
    │   ├── apps/         (rwxr-xr-x)
    │   └── workspace/    (rwxr-xr-x)
    ├── cache/            (rwx------)
    └── shared_prefs/     (rw-------)
```

### Network Security

- HTTPS for all downloads
- No cleartext traffic allowed

## Development

### Adding a New Application

Simply import any compatible JAR file through the Files tab. No code changes needed!

### JNI Development

The native layer is built with CMake:
```bash
cd app/src/main/cpp
cmake -B build -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26
```

## License

GNU General Public License v3.0 (GPL v3)

See [LICENSE](LICENSE) for full text.

## Acknowledgments

- [OpenJDK](https://openjdk.org) - Java Development Kit
- [Termux](https://termux.dev) - Android terminal environment
- [Termux:X11](https://github.com/termux/termux-x11) - X11 server for Android
- [Jetpack Compose](https://developer.android.com/jetpack/compose) - UI framework
- [OkHttp](https://square.github.io/okhttp/) - HTTP client

## Disclaimer

This is experimental software. Running Java applications on Android may cause:
- Increased battery consumption
- Device heating
- Performance degradation
- Unexpected crashes

Not all Java applications are compatible with Android's environment. Applications using:
- Windows/macOS/Linux native libraries
- JNI with platform-specific code
- Certain Java APIs not available in Android

...will not work. Use at your own risk.
