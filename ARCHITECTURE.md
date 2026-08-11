# Architecture Design Document

## Java Launcher for Android - Technical Architecture

### Overview

This document describes the architecture of the Java Launcher for Android, a universal Java application runner that enables executing JAR files on Android devices without Termux or manual configuration.

### Design Goals

1. **Universal Compatibility**: Support any Java application compatible with OpenJDK 17
2. **Zero Configuration**: First-time setup downloads and configures everything automatically
3. **Multiple GUI Modes**: Support headless, external X11, and future native bridge rendering
4. **Extensibility**: Easy to add new applications and rendering backends
5. **Performance**: Minimize overhead, leverage GPU where possible
6. **User-Provided Apps**: No bundled applications - users import their own JAR files

---

## System Architecture

### Layer Diagram

```
┌──────────────────────────────────────────────────────────────┐
│  Presentation Layer (Jetpack Compose)                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │  Home   │ │  Files  │ │  Logs   │ │ Settings│          │
│  │ Screen  │ │ Screen  │ │ Screen  │ │ Screen  │          │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘          │
│       └─────────────┴─────────┴─────────────┘              │
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
  │  - App import    │  - Progress flow  │    control    │
  │  - JAR import    │                   │               │
  └──────────────────────────────────────────────────────┘
        │
  ┌─────┴──────────────────────────────────────────────────┐
  │              Native Layer (JNI) - Optional            │
  │  ┌──────────────┐  ┌──────────────────────────────┐   │
  │  │ Swing Bridge │  │      VNC Client (RFB)        │   │
  │  │ (Future)     │  │  ANativeWindow + GPU texture │   │
  │  └──────────────┘  └──────────────────────────────┘   │
  └──────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. JavaAppManager

**Responsibility**: Central coordinator for JDK and application lifecycle

**Key Functions**:
- JDK installation verification and setup
- JAR file import from device storage (SAF)
- Application management (installed apps listing)
- Workspace directory management

**State Machine**:
```
Idle -> Checking -> DownloadingJdk -> Extracting -> Ready
                          |
                          v
                        Error (recoverable)
```

**Import Flow**:
```
User selects JAR via SAF
    |
    v
ContentResolver.openInputStream(uri)
    |
    v
Copy to /data/data/<pkg>/files/apps/<AppName>/<AppName>.jar
    |
    v
Validate (exists, non-empty)
    |
    v
Add to installed apps list
```

### 2. DownloadManager

**Responsibility**: Reliable file downloading with resume support

**Features**:
- HTTP Range requests for resume capability
- Progress reporting via StateFlow
- Concurrent download management
- Automatic retry on failure

**Architecture**:
```
User Request -> Check Cache -> HTTP HEAD (size) -> Range Request -> Stream Write
                    |                                      |
                    v                                      v
              File exists?                            Progress Update
              Calculate offset                        StateFlow emit
```

### 3. JarExecutor

**Responsibility**: Java process execution and log capture

**Process Model**:
```
ProcessBuilder
    ├── java binary (from JDK)
    ├── JVM args (-Xmx, -Djava.awt.headless, etc.)
    ├── -jar <app.jar>
    ├── app args
    └── Environment (JAVA_HOME, PATH, DISPLAY)

Output Capture:
    stdout ──┐
             ├── merged ──> BufferedReader ──> StateFlow<String>
    stderr ──┘
```

**Lifecycle**:
```
Idle -> Starting -> Running -> Stopping -> Exited/Error
```

### 4. VNC Subsystem (External Display)

**Architecture**:
```
Java App (Swing/AWT)
    |
    ├── DISPLAY=:0 (Termux:X11) or DISPLAY=<host>:<display>
    |
    v
External X11 Server (Termux:X11 or remote VNC)
    |
    ├── X11 protocol / RFB protocol
    |
    v
Android Screen (via Termux:X11 app or VNC client)
```

**Input Forwarding**:
```
Touch Event -> Gesture Recognition -> X11/VNC Events
    |
    ├── Tap -> Mouse Click (button 1)
    ├── Long Press -> Right Click (button 3)
    ├── Drag -> Mouse Drag
    ├── Pinch -> Scroll wheel
    └── Keyboard -> KeyEvent (keysym mapping)
```

---

## GUI Rendering Strategies

### Strategy 1: Headless Mode (Current)

**Approach**: Run Java with `-Djava.awt.headless=true`

**Pros**:
- No display server required
- Lowest overhead
- Works with all Java apps (that support headless)
- Simple and reliable

**Cons**:
- No GUI visible
- Only command-line output

**Use Cases**: Servers, batch processing, tools with CLI output

### Strategy 2: External X11 via Termux:X11 (Recommended)

**Approach**: Use Termux:X11 app as the display server

**Setup**:
```
1. Install Termux + Termux:X11 from F-Droid
2. Start Termux:X11
3. Java Launcher sets DISPLAY=:0
4. Java app renders to Termux:X11's X server
5. Termux:X11 displays on Android screen
```

**Pros**:
- Full GUI support
- No network overhead (Unix socket)
- Native Android integration
- Best performance for GUI apps

**Cons**:
- Requires separate app installation
- User must start Termux:X11 manually

**Use Cases**: Swing applications, IDEs, complex GUIs

### Strategy 3: External VNC Server

**Approach**: Connect to existing VNC server

**Architecture**:
```
Java App (Swing) -> Xvfb -> x11vnc (RFB) -> VncClient -> SurfaceView
```

**Pros**:
- Works remotely
- Standard protocol

**Cons**:
- Requires VNC server setup
- Network overhead

**Use Cases**: Remote access, headless servers with VNC

### Strategy 4: Native Bridge (Future)

**Approach**: Custom OpenJDK with JNI-integrated AWT peers

**Architecture**:
```
Java: Graphics2D.drawRect(x, y, w, h)
    |
    └── JNI call
        |
        └── nativeDrawRect(JNIEnv*, x, y, w, h, color)
            |
            └── ANativeWindow_Buffer
                |
                └── Direct pixel manipulation
                    |
                    └── GPU Composition
```

**Pros**:
- Zero network overhead
- Direct GPU rendering
- Lowest latency
- Best performance

**Cons**:
- Requires custom OpenJDK build
- Complex JNI implementation
- Maintenance burden
- Platform-specific

**Implementation Plan**:
1. Fork OpenJDK 17
2. Modify `sun.awt.X11Graphics` to call JNI instead of X11
3. Implement `AndroidGraphicsEnvironment`
4. Build Android-specific JDK

---

## Data Flow

### Application Import Flow

```
User taps "+" in Files tab
    |
    v
SAF: OpenDocument contract
    |
    v
User selects .jar file
    |
    v
FileBrowserViewModel.importJar(uri)
    |
    v
JavaAppManager.importJar(uri)
    |
    ├── ContentResolver.openInputStream(uri)
    ├── Read DISPLAY_NAME
    ├── Create /data/.../apps/<SafeName>/<SafeName>.jar
    ├── Copy stream
    └── Validate
    |
    v
Refresh app list
    |
    v
HomeScreen shows new app
```

### Application Launch Flow

```
User taps "Start"
    |
    v
HomeViewModel.startJavaApp()
    |
    v
JavaAppManager.getAppJar(name)
    |
    v
JarExecutor.startJar(
    javaBinary = /data/.../jdk/bin/java,
    jarFile = /data/.../apps/MyApp/MyApp.jar,
    jvmArgs = [-Xmx512m, -Djava.awt.headless=true],
    envVars = {JAVA_HOME: ..., PATH: ...}
)
    |
    ├── ProcessBuilder.start()
    |
    └── Coroutine: readProcessOutput()
            |
            └── BufferedReader.readLine()
                    |
                    └── StateFlow<String> logs
                            |
                            └── LogViewerScreen (auto-scroll)
```

### Setup Flow

```
User taps "Install JDK"
    |
    v
SetupViewModel.startSetup(url)
    |
    v
JavaAppManager.setupJdk(url)
    |
    ├── Check existing installation
    |       |
    |       └── Yes -> Ready
    |       |
    |       └── No -> Continue
    |
    ├── DownloadManager.download(jdkUrl, cacheFile)
    |       |
    |       └── Progress updates -> UI
    |
    ├── Extract ZIP to /data/.../files/jdk/
    |
    ├── Make binaries executable
    |
    └── Verify: java -version
            |
            └── Success -> Ready
            |
            └── Fail -> Error
```

---

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
- Certificate pinning for preset URLs (future)
- No cleartext traffic allowed (android:usesCleartextTraffic="false")

---

## Performance Considerations

### Memory Management

| Component | Memory Usage |
|-----------|-------------|
| Android App (UI) | ~50-100MB |
| OpenJDK 17 | ~150-300MB (base) |
| Java Application | Configurable (default 512MB max) |
| Termux:X11 (if used) | ~50MB |
| Total | ~750MB-1GB typical |

### Optimization Strategies

1. **Lazy Loading**: JDK components loaded on-demand
2. **Shared Libraries**: Use system fonts and timezone data
3. **Memory Mapping**: JAR files memory-mapped when possible
4. **GPU Rendering**: External X11 uses GPU composition
5. **Process Priority**: Background service with appropriate priority

---

## Extension Points

### Adding New GUI Backends

1. Implement `GuiBackend` interface:
```kotlin
interface GuiBackend {
    fun initialize(): Boolean
    fun renderFrame(buffer: ByteArray)
    fun sendInput(event: InputEvent)
    fun destroy()
}
```

2. Register in `JavaAppManager`
3. Add UI toggle in Settings

### Custom JVM Arguments

Users can configure per-app JVM arguments in Settings:
```
-Dproperty=value -Xspecialflag
```

---

## Testing Strategy

### Unit Tests

- DownloadManager: Mock HTTP server, resume logic
- JarExecutor: Mock Process, output parsing
- State machines: All state transitions

### Integration Tests

- Full setup flow with local HTTP server
- JAR import via SAF mock
- JAR execution with test applications

### Device Tests

- Memory usage under load
- Battery consumption
- Thermal throttling behavior
- Different Android versions (API 26-34)

---

## Future Roadmap

### Phase 1: Foundation (Current)
- [x] JDK download and setup
- [x] JAR import via SAF
- [x] JAR execution with headless mode
- [x] Process log capture
- [x] Basic file management

### Phase 2: GUI Support
- [ ] Termux:X11 integration
- [ ] VNC client with touch input mapping
- [ ] Virtual mouse and keyboard

### Phase 3: Native Bridge
- [ ] Custom OpenJDK build for Android
- [ ] JNI Graphics2D bridge
- [ ] GPU-accelerated Swing rendering

### Phase 4: Advanced Features
- [ ] Multi-window support
- [ ] Clipboard sharing
- [ ] File drag-and-drop
- [ ] Cloud sync for applications

---

## References

- [OpenJDK Port for Android](https://github.com/openjdk/mobile)
- [Termux Packages](https://github.com/termux/termux-packages)
- [Termux:X11](https://github.com/termux/termux-x11)
- [RFB Protocol Specification](https://github.com/rfbproto/rfbproto)
- [Android NDK - ANativeWindow](https://developer.android.com/ndk/reference/group/a-native-window)
