import Foundation

/// Errors surfaced to the UI. Messages are written for the person reading them.
enum BackendError: LocalizedError {
    case unreachable(underlying: String)
    case unauthorized
    case http(status: Int, detail: String)
    case decoding(String)

    var errorDescription: String? {
        switch self {
        case .unreachable:
            "Can't reach the transcription engine."
        case .unauthorized:
            "The app isn't authorised to talk to the engine. Restart it with ./run.sh."
        case .http(_, let detail):
            detail
        case .decoding(let detail):
            "The engine sent something unexpected (\(detail))."
        }
    }

    var recoverySuggestion: String? {
        switch self {
        case .unreachable:
            "Start it with ./run.sh in the whisper-mac folder, or use Engine ▸ Start."
        case .unauthorized:
            "The session token in ~/.whisper-mac/session.json is stale."
        default:
            nil
        }
    }

    /// True when retrying later might succeed on its own.
    var isTransient: Bool {
        if case .unreachable = self { return true }
        return false
    }
}

/// Thin async wrapper over the backend's HTTP API.
struct BackendClient {
    let session: SessionInfo
    private let urlSession: URLSession

    init(session: SessionInfo) {
        self.session = session
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 30
        configuration.waitsForConnectivity = false
        self.urlSession = URLSession(configuration: configuration)
    }

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return encoder
    }()

    // MARK: - Endpoints

    func health() async throws -> HealthDTO {
        try await send(path: "/health", method: "GET")
    }

    func configuration() async throws -> ConfigDTO {
        try await send(path: "/config", method: "GET")
    }

    func modelStatus() async throws -> ModelStatusDTO {
        try await send(path: "/models/status", method: "GET")
    }

    @discardableResult
    func preloadModel() async throws -> ModelStatusDTO {
        try await send(path: "/models/preload", method: "POST")
    }

    /// Transcribe a file where it already sits — nothing is copied or uploaded,
    /// which matters when the file is a 4 GB screen recording.
    func submit(_ body: LocalTranscribeBody) async throws -> JobDTO {
        try await send(path: "/transcribe/local", method: "POST", body: body)
    }

    func jobs() async throws -> [JobDTO] {
        let list: JobListDTO = try await send(path: "/jobs", method: "GET")
        return list.jobs
    }

    func job(id: String) async throws -> JobDTO {
        try await send(path: "/jobs/\(id)", method: "GET")
    }

    @discardableResult
    func cancel(id: String) async throws -> JobDTO {
        try await send(path: "/jobs/\(id)", method: "DELETE")
    }

    func transcript(id: String) async throws -> TranscriptDTO {
        try await send(path: "/jobs/\(id)/transcript", method: "GET")
    }

    // MARK: - Transport

    private func send<Response: Decodable>(path: String, method: String) async throws -> Response {
        try await perform(makeRequest(path: path, method: method))
    }

    private func send<Response: Decodable, Body: Encodable>(
        path: String,
        method: String,
        body: Body
    ) async throws -> Response {
        var request = makeRequest(path: path, method: method)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try Self.encoder.encode(body)
        return try await perform(request)
    }

    private func makeRequest(path: String, method: String) -> URLRequest {
        var request = URLRequest(url: session.url.appendingPathComponent(path))
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let token = session.token {
            request.setValue(token, forHTTPHeaderField: "X-Whisper-Token")
        }
        return request
    }

    private func perform<Response: Decodable>(_ request: URLRequest) async throws -> Response {
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await urlSession.data(for: request)
        } catch {
            throw BackendError.unreachable(underlying: error.localizedDescription)
        }

        guard let http = response as? HTTPURLResponse else {
            throw BackendError.decoding("no HTTP status")
        }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { throw BackendError.unauthorized }
            throw BackendError.http(
                status: http.statusCode,
                detail: Self.detail(from: data, status: http.statusCode)
            )
        }

        do {
            return try Self.decoder.decode(Response.self, from: data)
        } catch {
            throw BackendError.decoding(error.localizedDescription)
        }
    }

    /// FastAPI puts the readable message in `detail`, which may itself be a
    /// validation-error array.
    private static func detail(from data: Data, status: Int) -> String {
        if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let message = object["detail"] as? String { return message }
            if let items = object["detail"] as? [[String: Any]] {
                let messages = items.compactMap { $0["msg"] as? String }
                if !messages.isEmpty { return messages.joined(separator: "; ") }
            }
        }
        return "The engine returned HTTP \(status)."
    }
}
