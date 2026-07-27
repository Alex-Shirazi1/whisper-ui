import Foundation

// MARK: - Stage

/// Mirrors `backend.jobs.JobStage`, plus two client-only cases.
///
/// Decoding falls back to `.unknown` rather than throwing so a newer backend
/// can add stages without breaking an older app build.
enum JobStage: String, Codable, Hashable {
    /// Client-only: dropped but not submitted yet.
    case ready
    case queued
    case inspecting
    case extractingAudio = "extracting_audio"
    case downloadingModel = "downloading_model"
    case loadingModel = "loading_model"
    case transcribing
    case writingOutput = "writing_output"
    case completed
    case failed
    case cancelled
    case unknown

    init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = JobStage(rawValue: raw) ?? .unknown
    }

    var isTerminal: Bool {
        switch self {
        case .completed, .failed, .cancelled: true
        default: false
        }
    }

    var isRunning: Bool {
        switch self {
        case .ready, .completed, .failed, .cancelled, .unknown: false
        default: true
        }
    }

    /// Short label for the row badge.
    var label: String {
        switch self {
        case .ready: "Ready"
        case .queued: "Queued"
        case .inspecting: "Reading"
        case .extractingAudio: "Audio"
        case .downloadingModel: "Model"
        case .loadingModel: "Loading"
        case .transcribing: "Transcribing"
        case .writingOutput: "Writing"
        case .completed: "Done"
        case .failed: "Failed"
        case .cancelled: "Cancelled"
        case .unknown: "Working"
        }
    }

    var symbol: String {
        switch self {
        case .ready: "circle.dashed"
        case .queued: "clock"
        case .inspecting: "magnifyingglass"
        case .extractingAudio: "waveform"
        case .downloadingModel: "arrow.down.circle"
        case .loadingModel: "cpu"
        case .transcribing: "text.viewfinder"
        case .writingOutput: "square.and.arrow.down"
        case .completed: "checkmark.circle.fill"
        case .failed: "exclamationmark.triangle.fill"
        case .cancelled: "slash.circle"
        case .unknown: "circle"
        }
    }
}

// MARK: - Wire types

struct OutputFileDTO: Codable, Hashable {
    let format: String
    let path: String
    let filename: String

    var url: URL { URL(fileURLWithPath: path) }
    var outputFormat: OutputFormat? { OutputFormat(rawValue: format) }
}

struct JobDTO: Codable, Identifiable, Hashable {
    let id: String
    let name: String
    let stage: JobStage
    let statusMessage: String
    let progress: Double
    let indeterminate: Bool
    let sourcePath: String
    let outputDir: String
    let formats: [String]
    let model: String
    let task: String
    let outputs: [OutputFileDTO]
    let error: String?
    let mediaDuration: Double?
    let detectedLanguage: String?
    let isVideo: Bool
    let elapsedSeconds: Double
    let etaSeconds: Double?
    let speedFactor: Double?
    let createdAt: Double
    let finishedAt: Double?
}

struct JobListDTO: Codable {
    let jobs: [JobDTO]
}

struct EngineDTO: Codable, Hashable {
    let available: Bool
    let device: String?
    let mlxVersion: String?
    let mlxWhisperVersion: String?
    let error: String?
}

struct HealthDTO: Codable, Hashable {
    let status: String
    let version: String
    let model: String
    let engine: EngineDTO
    let ffmpeg: String?
    let modelCached: Bool
    let modelLoaded: Bool
    let activeJobs: Int
}

struct ConfigDTO: Codable, Hashable {
    let model: String
    let outputDir: String
    let formats: [String]
    let defaultFormats: [String]
    let supportedExtensions: [String]
    let maxUploadBytes: Int
    let wordTimestamps: Bool
    let temperature: Double
    let logPath: String
}

struct ModelStatusDTO: Codable, Hashable {
    let model: String
    let cached: Bool
    let loaded: Bool
    let cacheDir: String
    let preloading: Bool
    let error: String?
}

struct WordDTO: Codable, Hashable {
    let start: Double
    let end: Double
    let word: String
    let probability: Double?
}

struct SegmentDTO: Codable, Hashable, Identifiable {
    let id: Int
    let start: Double
    let end: Double
    let text: String
    let speaker: String?
    let words: [WordDTO]
}

struct TranscriptDTO: Codable, Hashable {
    let jobId: String
    let source: String
    let language: String?
    let duration: Double?
    let model: String
    let task: String
    let wordCount: Int
    let text: String
    let segments: [SegmentDTO]
}

struct LocalTranscribeBody: Encodable {
    let path: String
    let formats: [String]
    let outputDir: String?
    let model: String?
    let language: String?
    let task: String
    let initialPrompt: String?
    let overwrite: Bool
}

// MARK: - Queue item

/// One row in the app's queue: a dropped file, plus whatever the backend has
/// told us about it so far.
struct QueueItem: Identifiable, Hashable {
    let id: UUID
    let url: URL
    var jobID: String?
    var job: JobDTO?
    var localError: String?

    init(url: URL) {
        self.id = UUID()
        self.url = url
    }

    var name: String { url.lastPathComponent }

    var stage: JobStage {
        if localError != nil { return .failed }
        return job?.stage ?? .ready
    }

    var progress: Double { job?.progress ?? 0 }

    var statusMessage: String {
        if let localError { return localError }
        if let job { return job.statusMessage }
        return "Ready to transcribe"
    }

    var errorMessage: String? { localError ?? job?.error }

    var outputs: [OutputFileDTO] { job?.outputs ?? [] }

    var isActive: Bool { stage.isRunning }

    var canStart: Bool { jobID == nil && localError == nil }

    var canCancel: Bool { stage.isRunning }
}
