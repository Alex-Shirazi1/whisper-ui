import Foundation

/// The transcript formats the backend can write.
///
/// Order matches the backend's `OUTPUT_FORMATS` so the UI and the files on disk
/// always agree.
enum OutputFormat: String, CaseIterable, Identifiable, Codable, Hashable {
    case txt
    case srt
    case vtt
    case json

    var id: String { rawValue }

    var label: String { rawValue.uppercased() }

    var blurb: String {
        switch self {
        case .txt: "Plain text"
        case .srt: "Subtitles"
        case .vtt: "Web captions"
        case .json: "Segments + words"
        }
    }

    var symbol: String {
        switch self {
        case .txt: "doc.text"
        case .srt: "captions.bubble"
        case .vtt: "globe"
        case .json: "curlybraces"
        }
    }
}

extension Set<OutputFormat> {
    /// Stable, backend-ordered list for requests and display.
    var ordered: [OutputFormat] {
        OutputFormat.allCases.filter { contains($0) }
    }

    var rawValues: [String] {
        ordered.map(\.rawValue)
    }
}
