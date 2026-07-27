import Foundation

/// Starts and stops the Python backend on the app's behalf.
///
/// The point of this app is never needing a terminal, so double-clicking
/// WhisperMac.app has to be enough. `scripts/build.sh` stamps the repo path into
/// the bundle's Info.plist; if the app was moved somewhere the repo can't be
/// found, the UI falls back to asking the user to run `./run.sh`.
final class BackendLauncher {
    enum LaunchError: LocalizedError {
        case repositoryNotFound
        case scriptMissing(URL)
        case launchFailed(String)

        var errorDescription: String? {
            switch self {
            case .repositoryNotFound:
                "WhisperMac can't find its whisper-mac folder."
            case .scriptMissing(let url):
                "Missing launcher script at \(url.path)."
            case .launchFailed(let detail):
                "The engine failed to start: \(detail)"
            }
        }

        var recoverySuggestion: String? {
            switch self {
            case .repositoryNotFound, .scriptMissing:
                "Open Terminal in the whisper-mac folder and run ./run.sh once, or rebuild with ./scripts/build.sh."
            case .launchFailed:
                "Check logs/whisper-mac.log for details."
            }
        }
    }

    private var process: Process?
    private let fileManager = FileManager.default

    /// True when this app owns the running backend and should stop it on quit.
    var isManagingBackend: Bool { process?.isRunning == true }

    // MARK: - Repo discovery

    /// Candidate locations, most trustworthy first.
    func repositoryRoot() -> URL? {
        var candidates: [URL] = []

        if let path = ProcessInfo.processInfo.environment["WHISPER_MAC_REPO"], !path.isEmpty {
            candidates.append(URL(fileURLWithPath: path))
        }
        if let stamped = Bundle.main.object(forInfoDictionaryKey: "WMRepoRoot") as? String,
           !stamped.isEmpty {
            candidates.append(URL(fileURLWithPath: stamped))
        }
        // build/WhisperMac.app -> repo root
        candidates.append(
            Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent()
        )

        return candidates.first(where: isRepository)
    }

    private func isRepository(_ url: URL) -> Bool {
        fileManager.fileExists(atPath: url.appendingPathComponent("scripts/run_backend.sh").path)
            && fileManager.fileExists(atPath: url.appendingPathComponent("backend/whisper_server.py").path)
    }

    // MARK: - Lifecycle

    @discardableResult
    func start() throws -> URL {
        if isManagingBackend { return try requireRepository() }

        let root = try requireRepository()
        let script = root.appendingPathComponent("scripts/run_backend.sh")
        guard fileManager.isExecutableFile(atPath: script.path) else {
            throw LaunchError.scriptMissing(script)
        }

        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/bash")
        task.arguments = [script.path]
        task.currentDirectoryURL = root

        // A GUI process inherits a bare PATH; put Homebrew back so the script
        // and ffmpeg are both findable.
        var environment = ProcessInfo.processInfo.environment
        let existingPath = environment["PATH"] ?? "/usr/bin:/bin"
        environment["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + existingPath
        environment["WHISPER_MAC_LAUNCHED_BY_APP"] = "1"
        task.environment = environment

        // Output goes to the rotating log file the script sets up.
        task.standardOutput = FileHandle.nullDevice
        task.standardError = FileHandle.nullDevice

        do {
            try task.run()
        } catch {
            throw LaunchError.launchFailed(error.localizedDescription)
        }
        process = task
        return root
    }

    func stop() {
        guard let process, process.isRunning else { return }
        process.terminate()
        self.process = nil
    }

    private func requireRepository() throws -> URL {
        guard let root = repositoryRoot() else { throw LaunchError.repositoryNotFound }
        return root
    }
}
