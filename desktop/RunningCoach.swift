import AppKit
import WebKit

final class CoachApp: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate, WKUIDelegate {
    private var window: NSWindow!
    private var web: WKWebView!
    private var backend: Process?
    private var readyFile: URL!
    private var workspace: URL!
    private var baseURL: URL?
    private var timer: Timer?
    private var shuttingDown = false
    private var launchTime = Date()
    private var statusLabel: NSTextField!
    private var activity: NSObjectProtocol?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let ownID = Bundle.main.bundleIdentifier ?? "com.gabemurray.runningcoach"
        if let other = NSRunningApplication.runningApplications(withBundleIdentifier: ownID).first(where: {$0.processIdentifier != ProcessInfo.processInfo.processIdentifier}) {
            other.activate(options: [.activateAllWindows])
            NSApp.terminate(nil)
            return
        }
        let fm = FileManager.default
        let workspaceName = Bundle.main.object(forInfoDictionaryKey: "RunningCoachWorkspace") as? String ?? "Running Coach"
        workspace = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent(workspaceName, isDirectory: true)
        // Explicit testing override. Never read a developer checkout implicitly.
        if let i = CommandLine.arguments.firstIndex(of: "--data-dir"), CommandLine.arguments.count > i+1 {
            workspace = URL(fileURLWithPath: CommandLine.arguments[i+1], isDirectory: true)
        }
        do {
            try fm.createDirectory(at: workspace, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        } catch { fail("Could not open your app-data folder: \(error.localizedDescription)"); return }
        setupMenu()
        let config = WKWebViewConfiguration()
        web = WKWebView(frame: .zero, configuration: config)
        web.navigationDelegate = self
        web.uiDelegate = self
        web.allowsBackForwardNavigationGestures = true
        web.setValue(false, forKey: "drawsBackground")
        window = NSWindow(contentRect: NSRect(x:0,y:0,width:1440,height:950), styleMask:[.titled,.closable,.miniaturizable,.resizable], backing:.buffered, defer:false)
        window.title = Bundle.main.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String ?? "Running Coach"
        window.minSize = NSSize(width:720,height:550)
        window.backgroundColor = NSColor(calibratedRed:0.035,green:0.055,blue:0.075,alpha:1)
        window.delegate = self
        window.isReleasedWhenClosed = false
        window.setFrameAutosaveName(ownID)
        window.contentView = web
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        showStatus("Starting Running Coach…")
        launchBackend()
    }

    private func setupMenu() {
        let menu = NSMenu()
        let appItem = NSMenuItem()
        let app = NSMenu()
        app.addItem(withTitle:"About Running Coach",action:#selector(about),keyEquivalent:"")
        app.addItem(.separator())
        app.addItem(withTitle:"Open Data Folder",action:#selector(openData),keyEquivalent:"")
        app.addItem(withTitle:"Third-Party Licenses",action:#selector(openLicenses),keyEquivalent:"")
        app.addItem(.separator())
        app.addItem(withTitle:"Hide Running Coach",action:#selector(NSApplication.hide(_:)),keyEquivalent:"h")
        app.addItem(withTitle:"Quit Running Coach",action:#selector(NSApplication.terminate(_:)),keyEquivalent:"q")
        appItem.submenu = app
        menu.addItem(appItem)
        let editItem = NSMenuItem(); let edit = NSMenu(title:"Edit")
        for (title,selector,key) in [("Undo","undo:","z"),("Cut","cut:","x"),("Copy","copy:","c"),("Paste","paste:","v"),("Select All","selectAll:","a")] {
            edit.addItem(withTitle:title,action:NSSelectorFromString(selector),keyEquivalent:key)
        }
        editItem.submenu = edit; menu.addItem(editItem)
        let viewItem = NSMenuItem(); let view = NSMenu(title:"View")
        view.addItem(withTitle:"Reload",action:#selector(reload),keyEquivalent:"r")
        view.addItem(withTitle:"Open in Browser",action:#selector(openBrowser),keyEquivalent:"")
        viewItem.submenu = view; menu.addItem(viewItem)
        let actions = [#selector(about), #selector(openData), #selector(openLicenses), #selector(reload), #selector(openBrowser)]
        for item in app.items + view.items {
            if let action = item.action, actions.contains(action) {item.target = self}
        }
        NSApp.mainMenu = menu
    }

    private func showStatus(_ text: String) {
        statusLabel?.removeFromSuperview()
        statusLabel = NSTextField(labelWithString:text)
        statusLabel.textColor = NSColor(calibratedRed:0.47,green:0.83,blue:0.70,alpha:1)
        statusLabel.font = NSFont.systemFont(ofSize:20,weight:.semibold)
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        web.addSubview(statusLabel)
        NSLayoutConstraint.activate([statusLabel.centerXAnchor.constraint(equalTo:web.centerXAnchor),statusLabel.centerYAnchor.constraint(equalTo:web.centerYAnchor)])
    }

    private func launchBackend() {
        guard let resources = Bundle.main.resourceURL else { fail("The app bundle is incomplete."); return }
        let executable = resources.appendingPathComponent("backend/RunningCoachServer")
        readyFile = workspace.appendingPathComponent(".launch-\(UUID().uuidString).json")
        let process = Process()
        process.executableURL = executable
        process.arguments = ["--data-dir",workspace.path,"--ready-file",readyFile.path,"--parent-pid",String(ProcessInfo.processInfo.processIdentifier)]
        process.currentDirectoryURL = workspace
        var env = ProcessInfo.processInfo.environment
        for key in ["PYTHONHOME","PYTHONPATH","DYLD_LIBRARY_PATH","DYLD_FRAMEWORK_PATH"] {env.removeValue(forKey:key)}
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env
        // Python writes a rotating private log. A second launch cannot fill a
        // pipe and deadlock while a long planner search is running.
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        backend = process
        do {try process.run()} catch {fail("Could not start the local engine: \(error.localizedDescription)"); return}
        activity = ProcessInfo.processInfo.beginActivity(options:[.userInitiatedAllowingIdleSystemSleep],reason:"Running Coach local training analysis")
        launchTime = Date()
        timer = Timer.scheduledTimer(withTimeInterval:0.15,repeats:true) { [weak self] _ in self?.checkStartup() }
    }

    private func checkStartup() {
        guard let process = backend else {return}
        if !process.isRunning {
            timer?.invalidate()
            fail("The local engine stopped. Details are in the app-data folder’s logs/app.log. Your data has not been deleted.")
            return
        }
        if let bytes = try? Data(contentsOf:readyFile), let value = try? JSONSerialization.jsonObject(with:bytes) as? [String:Any],
           let address = value["url"] as? String, let url = URL(string:address), url.host == "127.0.0.1", url.scheme == "http" {
            baseURL = url
            timer?.invalidate()
            web.load(URLRequest(url:url))
            return
        }
        if Date().timeIntervalSince(launchTime)>120 {
            timer?.invalidate()
            fail("Starting took longer than expected. Quit and reopen the app. Details are in logs/app.log.")
        }
    }

    private func fail(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "Running Coach needs attention"
        alert.informativeText = message
        alert.addButton(withTitle:"Close")
        if workspace != nil {alert.addButton(withTitle:"Open Data Folder")}
        if alert.runModal() == .alertSecondButtonReturn {openData()}
        NSApp.terminate(nil)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {statusLabel?.removeFromSuperview()}
    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy)->Void) {
        guard let url = navigationAction.request.url else {decisionHandler(.cancel); return}
        if url.scheme == "about" || url.scheme == "blob" || (url.host == baseURL?.host && url.port == baseURL?.port && url.scheme == "http") {
            decisionHandler(.allow)
        } else {
            if ["https","http","mailto"].contains(url.scheme ?? "") {NSWorkspace.shared.open(url)}
            decisionHandler(.cancel)
        }
    }
    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration, for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures)->WKWebView? {
        if let url = navigationAction.request.url, ["https","http"].contains(url.scheme ?? "") {NSWorkspace.shared.open(url)}
        return nil
    }
    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?)->Void) {
        let panel = NSOpenPanel(); panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = false; panel.canChooseFiles = true
        panel.beginSheetModal(for:window) {result in completionHandler(result == .OK ? panel.urls : nil)}
    }
    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message:String, initiatedByFrame frame:WKFrameInfo, completionHandler:@escaping ()->Void) {
        let alert = NSAlert(); alert.messageText = message
        alert.beginSheetModal(for:window) {_ in completionHandler()}
    }
    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message:String, initiatedByFrame frame:WKFrameInfo, completionHandler:@escaping (Bool)->Void) {
        let alert = NSAlert(); alert.messageText = message
        alert.addButton(withTitle:"Continue"); alert.addButton(withTitle:"Cancel")
        alert.beginSheetModal(for:window) {result in completionHandler(result == .alertFirstButtonReturn)}
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender:NSApplication)->Bool {true}
    func applicationShouldTerminate(_ sender:NSApplication)->NSApplication.TerminateReply {
        guard let process = backend, process.isRunning else {return .terminateNow}
        if shuttingDown {return .terminateLater}
        shuttingDown = true; timer?.invalidate()
        showStatus("Closing Running Coach — finishing any active calculation…")
        process.terminationHandler = { _ in DispatchQueue.main.async {NSApp.reply(toApplicationShouldTerminate:true)} }
        process.terminate() // Uvicorn finishes in-flight writes and jobs cleanly.
        return .terminateLater
    }
    func applicationWillTerminate(_ notification:Notification) {
        if let item = activity {ProcessInfo.processInfo.endActivity(item)}
        if let path = readyFile {try? FileManager.default.removeItem(at:path)}
    }
    @objc private func reload() {web?.reloadFromOrigin()}
    @objc private func openData() {if let path = workspace {NSWorkspace.shared.open(path)}}
    @objc private func openBrowser() {if let address = web?.url ?? baseURL {NSWorkspace.shared.open(address)}}
    @objc private func openLicenses() {if let path = Bundle.main.resourceURL?.appendingPathComponent("licenses") {NSWorkspace.shared.open(path)}}
    @objc private func about() {NSApp.orderFrontStandardAboutPanel(options:[.applicationName:"Running Coach",.applicationVersion:"0.2.0"])}
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
app.appearance = NSAppearance(named:.darkAqua)
let delegate = CoachApp()
app.delegate = delegate
app.run()
