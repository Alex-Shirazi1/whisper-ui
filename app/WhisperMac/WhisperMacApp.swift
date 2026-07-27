import AppKit
import SwiftUI

@main
struct WhisperMacApp: App {
    @StateObject private var model = AppModel()
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup("WhisperMac") {
            ContentView()
                .environmentObject(model)
                .onAppear {
                    // The delegate owns termination cleanup; it needs the model
                    // that actually holds the running backend.
                    appDelegate.model = model
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentMinSize)
        .defaultSize(width: 1080, height: 720)
        .commands { WhisperCommands() }
    }
}

// MARK: - Menu bar

struct WhisperCommands: Commands {
    @FocusedValue(\.appModel) private var model

    var body: some Commands {
        // "New Window" makes no sense for a single-queue app.
        CommandGroup(replacing: .newItem) {}

        CommandGroup(after: .newItem) {
            Button("Add Files…") { model?.chooseFiles() }
                .keyboardShortcut("o", modifiers: .command)
                .disabled(model == nil)
        }

        CommandMenu("Transcribe") {
            Button("Start Queued Files") { model?.startAll() }
                .keyboardShortcut("r", modifiers: .command)
                .disabled(model?.canStart != true)

            Button("Cancel All") { model?.cancelAll() }
                .keyboardShortcut(".", modifiers: .command)
                .disabled(model?.hasWork != true)

            Divider()

            Button("Clear Finished") { model?.clearFinished() }
                .keyboardShortcut("k", modifiers: [.command, .shift])
                .disabled(model?.finishedItems.isEmpty != false)

            Divider()

            Button("Copy Transcript") { model?.copyTranscriptToClipboard() }
                .keyboardShortcut("c", modifiers: [.command, .shift])
                .disabled(model?.transcript == nil)
        }

        CommandMenu("Engine") {
            Button("Start Engine") { model?.startEngine() }
                .disabled(model == nil || model?.connection.isOnline == true)

            Button("Reconnect") { Task { await model?.refreshHealth() } }
                .keyboardShortcut("r", modifiers: [.command, .shift])
        }

        CommandGroup(after: .saveItem) {
            Button("Open Output Folder") { model?.openOutputFolder() }
                .keyboardShortcut("o", modifiers: [.command, .shift])
            Button("Choose Output Folder…") { model?.chooseOutputFolder() }
        }
    }
}

// MARK: - App delegate

/// Handles the two things SwiftUI can't: closing the window should quit, and
/// quitting should take a backend we started with us.
final class AppDelegate: NSObject, NSApplicationDelegate {
    var model: AppModel?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    /// Finder's "Open With", a Dock drop, or `open -a WhisperMac file.mov`.
    /// The Info.plist advertises audio/video document types, so this has to work
    /// as well as an in-window drop does.
    func application(_ application: NSApplication, open urls: [URL]) {
        MainActor.assumeIsolated {
            model?.add(urls: urls)
        }
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated {
            model?.stopMonitoring()
            model?.stopEngineIfManaged()
        }
    }
}
