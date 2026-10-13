import AppKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers

/// Single source of truth for the window: the queue, the connection, and the
/// user's choices. Views read it and call intents on it; nothing else mutates
/// state.
@MainActor
final class AppModel: ObservableObject {

    enum ConnectionState: Equatable {
        case connecting
        case online
        case offline(String)

        var isOnline: Bool { self == .online }
    }

    // MARK: - Published state

    @Published private(set) var items: [QueueItem] = []
    @Published var selection: UUID?
    @Published private(set) var transcript: TranscriptDTO?
    @Published private(set) var isLoadingTranscript = false

    @Published private(set) var connection: ConnectionState = .connecting
    @Published private(set) var health: HealthDTO?
    @Published private(set) var isStartingEngine = false
    /// Transient, user-facing problem (submission failed, cancel failed, …).
    @Published var notice: String?

    @Published var formats: Set<OutputFormat> {
        didSet { Preferences.formats = formats }
    }
    @Published var outputFolder: URL {
        didSet {
            if hasChosenFolder { Preferences.outputFolder = outputFolder }
        }
    }
    @Published var language: LanguageOption {
        didSet { Preferences.language = language }
    }
    @Published var translateToEnglish: Bool {
        didSet { Preferences.translateToEnglish = translateToEnglish }
    }
    @Published var autoStart: Bool {
        didSet { Preferences.autoStart = autoStart }
    }
    @Published var revealWhenDone: Bool {
        didSet { Preferences.revealWhenDone = revealWhenDone }
    }

    // MARK: - Private

    private var client: BackendClient
    private let launcher = BackendLauncher()
    private var monitor: Task<Void, Never>?
    private var hasChosenFolder: Bool
    private var didAdoptBackendDefaults = false
    private var supportedExtensions: Set<String> = AppModel.builtInExtensions

    /// Mirrors `backend/config.py`; replaced by `/config` once connected.
    static let builtInExtensions: Set<String> = [
        "mp4", "mov", "m4v", "mkv", "webm", "avi", "mpg", "mpeg", "wmv", "flv",
        "m4a", "wav", "mp3", "aac", "flac", "ogg", "oga", "opus", "aiff", "aif", "wma", "caf",
    ]

    // MARK: - Init

    init() {
        let storedFolder = Preferences.outputFolder
        hasChosenFolder = storedFolder != nil
        formats = Preferences.formats
        outputFolder = storedFolder
            ?? SessionStore.suggestedOutputFolder()
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Documents/WhisperMac", isDirectory: true)
        language = Preferences.language
        translateToEnglish = Preferences.translateToEnglish
        autoStart = Preferences.autoStart
        revealWhenDone = Preferences.revealWhenDone
        client = BackendClient(session: SessionStore.current())
    }

    // MARK: - Derived

    var activeItems: [QueueItem] { items.filter(\.isActive) }
    var waitingItems: [QueueItem] { items.filter(\.canStart) }
    var finishedItems: [QueueItem] { items.filter { $0.stage.isTerminal } }
    var hasWork: Bool { !activeItems.isEmpty }
    var canStart: Bool { !waitingItems.isEmpty && connection.isOnline }

    var selectedItem: QueueItem? {
        guard let selection else { return nil }
        return items.first { $0.id == selection }
    }

    /// Aggregate progress for the window-level bar.
    var overallProgress: Double {
        let running = items.filter { $0.stage.isRunning || $0.stage == .completed }
        guard !running.isEmpty else { return 0 }
        return running.reduce(0.0) { $0 + ($1.stage == .completed ? 1 : $1.progress) }
            / Double(running.count)
    }

    var headlineStatus: String {
        if case .offline = connection { return "Engine offline" }
        if let active = activeItems.first {
            return activeItems.count > 1
                ? "\(active.statusMessage)  ·  \(activeItems.count) files in flight"
                : active.statusMessage
        }
        if !waitingItems.isEmpty {
            return "\(waitingItems.count) file\(waitingItems.count == 1 ? "" : "s") ready"
        }
        if let last = finishedItems.last, last.stage == .completed {
            return "Finished — \(last.name)"
        }
        return "Ready"
    }

    var modelLabel: String {
        let name = health?.model ?? "parakeet-tdt-0.6b-v3"
        return name.split(separator: "/").last.map(String.init) ?? name
    }

    // MARK: - Monitoring

    func startMonitoring() {
        guard monitor == nil else { return }
        monitor = Task { [weak self] in
            var tick = 0
            while !Task.isCancelled {
                guard let self else { return }
                let online = self.connection.isOnline
                if tick % 4 == 0 || !online {
                    await self.refreshHealth()
                }
                if self.hasWork || tick % 8 == 0 {
                    await self.refreshJobs()
                }
                tick += 1
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }
    }

    func stopMonitoring() {
        monitor?.cancel()
        monitor = nil
    }

    func refreshHealth() async {
        do {
            let report = try await client.health()
            health = report
            connection = .online
            if !didAdoptBackendDefaults {
                didAdoptBackendDefaults = true
                await adoptBackendDefaults()
            }
        } catch {
            health = nil
            connection = .offline((error as? BackendError)?.errorDescription ?? error.localizedDescription)
            // The backend may have restarted on another port or rotated its token.
            let fresh = SessionStore.current()
            if fresh != client.session {
                client = BackendClient(session: fresh)
            }
        }
    }

    private func adoptBackendDefaults() async {
        guard let configuration = try? await client.configuration() else { return }
        supportedExtensions = Set(configuration.supportedExtensions.map {
            $0.hasPrefix(".") ? String($0.dropFirst()) : $0
        })
        if !hasChosenFolder {
            outputFolder = URL(fileURLWithPath: configuration.outputDir)
        }
        // Warm the weights so the first drop doesn't wait on a cold load.
        Task { try? await client.preloadModel() }
    }

    func refreshJobs() async {
        guard connection.isOnline else { return }
        guard let remote = try? await client.jobs() else { return }
        let byID = Dictionary(uniqueKeysWithValues: remote.map { ($0.id, $0) })

        var newlyCompleted: [QueueItem] = []
        items = items.map { item in
            guard let jobID = item.jobID, let fresh = byID[jobID] else { return item }
            var updated = item
            let previousStage = item.stage
            updated.job = fresh
            if previousStage != .completed, fresh.stage == .completed {
                newlyCompleted.append(updated)
            }
            return updated
        }

        if revealWhenDone, let finished = newlyCompleted.last,
           let first = finished.outputs.first {
            reveal(first.url)
        }
        if let selection, let item = items.first(where: { $0.id == selection }),
           item.stage == .completed, transcript?.jobId != item.jobID {
            await loadTranscript(for: item)
        }
    }

    // MARK: - Queue intents

    /// Add dropped or picked files, skipping unsupported types and duplicates.
    func add(urls: [URL]) {
        var rejected: [String] = []
        var added = false

        for url in urls {
            let resolved = url.resolvingSymlinksInPath()
            guard isSupported(resolved) else {
                rejected.append(resolved.lastPathComponent)
                continue
            }
            let alreadyQueued = items.contains {
                $0.url == resolved && !$0.stage.isTerminal
            }
            if alreadyQueued { continue }
            items.append(QueueItem(url: resolved))
            added = true
        }

        if !rejected.isEmpty {
            notice = rejected.count == 1
                ? "\(rejected[0]) isn't an audio or video file WhisperMac can read."
                : "\(rejected.count) files were skipped — unsupported formats."
        } else if added {
            notice = nil
        }

        if added, autoStart {
            startAll()
        }
        if selection == nil { selection = items.last?.id }
    }

    func isSupported(_ url: URL) -> Bool {
        supportedExtensions.contains(url.pathExtension.lowercased())
    }

    func startAll() {
        for item in waitingItems {
            Task { await submit(item) }
        }
    }

    func submit(_ item: QueueItem) async {
        guard let index = items.firstIndex(where: { $0.id == item.id }), items[index].canStart else {
            return
        }
        let body = LocalTranscribeBody(
            path: item.url.path(percentEncoded: false),
            formats: formats.rawValues,
            outputDir: outputFolder.path(percentEncoded: false),
            model: nil,
            language: language.requestValue,
            task: translateToEnglish ? "translate" : "transcribe",
            initialPrompt: nil,
            overwrite: false
        )

        do {
            let job = try await client.submit(body)
            guard let current = items.firstIndex(where: { $0.id == item.id }) else { return }
            items[current].jobID = job.id
            items[current].job = job
            items[current].localError = nil
        } catch {
            guard let current = items.firstIndex(where: { $0.id == item.id }) else { return }
            let message = (error as? BackendError)?.errorDescription ?? error.localizedDescription
            items[current].localError = message
            notice = message
        }
    }

    func cancel(_ item: QueueItem) {
        guard let jobID = item.jobID else {
            remove(item)
            return
        }
        Task {
            do {
                let job = try await client.cancel(id: jobID)
                if let index = items.firstIndex(where: { $0.id == item.id }) {
                    items[index].job = job
                }
            } catch {
                notice = (error as? BackendError)?.errorDescription ?? error.localizedDescription
            }
        }
    }

    func cancelAll() {
        activeItems.forEach(cancel)
    }

    func retry(_ item: QueueItem) {
        guard let index = items.firstIndex(where: { $0.id == item.id }) else { return }
        items[index].jobID = nil
        items[index].job = nil
        items[index].localError = nil
        let refreshed = items[index]
        Task { await submit(refreshed) }
    }

    func remove(_ item: QueueItem) {
        if item.canCancel, let jobID = item.jobID {
            Task { try? await client.cancel(id: jobID) }
        }
        items.removeAll { $0.id == item.id }
        if selection == item.id {
            selection = items.last?.id
            transcript = nil
        }
    }

    func clearFinished() {
        let survivors = items.filter { !$0.stage.isTerminal }
        items = survivors
        if let selection, !survivors.contains(where: { $0.id == selection }) {
            self.selection = survivors.last?.id
            transcript = nil
        }
    }

    // MARK: - Transcript

    func select(_ item: QueueItem) {
        selection = item.id
        transcript = nil
        guard item.stage == .completed else { return }
        Task { await loadTranscript(for: item) }
    }

    func loadTranscript(for item: QueueItem) async {
        guard let jobID = item.jobID else { return }
        isLoadingTranscript = true
        defer { isLoadingTranscript = false }
        transcript = try? await client.transcript(id: jobID)
    }

    // MARK: - Engine control

    func startEngine() {
        guard !isStartingEngine else { return }
        isStartingEngine = true
        notice = nil
        Task {
            defer { isStartingEngine = false }
            do {
                try launcher.start()
            } catch {
                notice = [
                    (error as? LocalizedError)?.errorDescription,
                    (error as? LocalizedError)?.recoverySuggestion,
                ]
                .compactMap { $0 }
                .joined(separator: " ")
                return
            }
            // Give uvicorn a moment, then re-point at whatever session it wrote.
            for _ in 0..<40 {
                try? await Task.sleep(nanoseconds: 500_000_000)
                client = BackendClient(session: SessionStore.current())
                await refreshHealth()
                if connection.isOnline { return }
            }
        }
    }

    func stopEngineIfManaged() {
        launcher.stop()
    }

    var canStartEngine: Bool { launcher.repositoryRoot() != nil }

    // MARK: - Finder / panels

    func openOutputFolder() {
        try? FileManager.default.createDirectory(
            at: outputFolder, withIntermediateDirectories: true
        )
        NSWorkspace.shared.open(outputFolder)
    }

    func reveal(_ url: URL) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func open(_ url: URL) {
        NSWorkspace.shared.open(url)
    }

    func copyTranscriptToClipboard() {
        guard let text = transcript?.text, !text.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        notice = "Transcript copied to the clipboard."
    }

    func chooseFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.message = "Choose audio or video files to transcribe"
        panel.prompt = "Add"
        panel.allowedContentTypes = supportedExtensions.compactMap {
            UTType(filenameExtension: $0)
        }
        if panel.runModal() == .OK {
            add(urls: panel.urls)
        }
    }

    func chooseOutputFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Where should transcripts be saved?"
        panel.prompt = "Choose"
        panel.directoryURL = outputFolder
        if panel.runModal() == .OK, let url = panel.url {
            hasChosenFolder = true
            outputFolder = url
            Preferences.outputFolder = url
        }
    }

    func toggle(_ format: OutputFormat) {
        var updated = formats
        if updated.contains(format) {
            // Always leave at least one format selected.
            guard updated.count > 1 else { return }
            updated.remove(format)
        } else {
            updated.insert(format)
        }
        formats = updated
    }
}
