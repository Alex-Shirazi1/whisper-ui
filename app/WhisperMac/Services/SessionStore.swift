import Foundation

/// Where the backend lives and how to authenticate with it.
struct SessionInfo: Equatable {
    let url: URL
    let token: String?

    static let fallback = SessionInfo(
        url: URL(string: "http://127.0.0.1:8756")!,
        token: nil
    )
}

/// Reads the handshake file the backend writes on boot.
///
/// This is what makes "double-click the app" work: the user never sees a port
/// or a token, and a backend started on a different port is still found.
enum SessionStore {
    static let stateDirectory = FileManager.default
        .homeDirectoryForCurrentUser
        .appendingPathComponent(".whisper-mac", isDirectory: true)

    static var sessionFile: URL {
        stateDirectory.appendingPathComponent("session.json")
    }

    private struct Payload: Decodable {
        let url: String
        let token: String?
        let pid: Int?
        let model: String?
        let outputDir: String?
        let version: String?
    }

    /// Environment overrides win, then the session file, then the default port.
    static func current() -> SessionInfo {
        let environment = ProcessInfo.processInfo.environment
        let envURL = environment["WHISPER_MAC_URL"].flatMap(URL.init(string:))
        let envToken = environment["WHISPER_MAC_TOKEN"].flatMap { $0.isEmpty ? nil : $0 }

        guard let payload = readPayload() else {
            return SessionInfo(url: envURL ?? SessionInfo.fallback.url, token: envToken)
        }
        return SessionInfo(
            url: envURL ?? URL(string: payload.url) ?? SessionInfo.fallback.url,
            token: envToken ?? payload.token
        )
    }

    /// Default output folder advertised by the running backend, if any.
    static func suggestedOutputFolder() -> URL? {
        guard let path = readPayload()?.outputDir, !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path)
    }

    private static func readPayload() -> Payload? {
        guard let data = try? Data(contentsOf: sessionFile) else { return nil }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try? decoder.decode(Payload.self, from: data)
    }
}
