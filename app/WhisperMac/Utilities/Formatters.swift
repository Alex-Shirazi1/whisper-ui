import Foundation

/// Display helpers. Kept in one place so a duration reads the same everywhere.
enum Format {
    /// `1:04` / `1:02:03` — for media lengths and elapsed time.
    static func clock(_ seconds: Double?) -> String {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return "--:--" }
        let total = Int(seconds.rounded())
        let (hours, minutes, secs) = (total / 3600, (total % 3600) / 60, total % 60)
        return hours > 0
            ? String(format: "%d:%02d:%02d", hours, minutes, secs)
            : String(format: "%d:%02d", minutes, secs)
    }

    /// `00:01:02.480` — subtitle-style, for the transcript inspector.
    static func timecode(_ seconds: Double) -> String {
        let clamped = max(0, seconds)
        let milliseconds = Int((clamped * 1000).rounded())
        let hours = milliseconds / 3_600_000
        let minutes = (milliseconds % 3_600_000) / 60_000
        let secs = (milliseconds % 60_000) / 1000
        return String(format: "%02d:%02d:%02d.%03d", hours, minutes, secs, milliseconds % 1000)
    }

    /// Rough, friendly remaining time.
    static func remaining(_ seconds: Double?) -> String? {
        guard let seconds, seconds.isFinite, seconds > 1 else { return nil }
        if seconds < 60 { return "about \(Int(seconds.rounded()))s left" }
        let minutes = Int((seconds / 60).rounded())
        return minutes <= 1 ? "about a minute left" : "about \(minutes) min left"
    }

    static func percent(_ fraction: Double) -> String {
        "\(Int((max(0, min(1, fraction)) * 100).rounded()))%"
    }

    static func speed(_ factor: Double?) -> String? {
        guard let factor, factor.isFinite, factor > 0 else { return nil }
        return String(format: "%.1f× realtime", factor)
    }

    static func fileSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useKB, .useMB, .useGB]
        return formatter.string(fromByteCount: bytes)
    }

    /// Shorten a path for display: `~/Movies/Talks`.
    static func shortPath(_ url: URL) -> String {
        let path = url.path(percentEncoded: false)
        let home = FileManager.default.homeDirectoryForCurrentUser.path(percentEncoded: false)
        return path.hasPrefix(home) ? "~" + path.dropFirst(home.count - 1) : path
    }

    static func languageName(_ code: String?) -> String? {
        guard let code, !code.isEmpty else { return nil }
        return Locale.current.localizedString(forLanguageCode: code)?.capitalized ?? code.uppercased()
    }
}

/// Languages offered in the picker. Whisper supports many more; these are the
/// ones worth a click, with auto-detect as the default.
enum LanguageOption: Hashable, Identifiable {
    case auto
    case code(String)

    var id: String {
        switch self {
        case .auto: "auto"
        case .code(let value): value
        }
    }

    var label: String {
        switch self {
        case .auto: "Auto-detect"
        case .code(let value): Format.languageName(value) ?? value
        }
    }

    /// Value sent to the backend (`nil` means detect).
    var requestValue: String? {
        switch self {
        case .auto: nil
        case .code(let value): value
        }
    }

    static let common: [LanguageOption] = [
        .auto,
        .code("en"), .code("es"), .code("fr"), .code("de"), .code("it"),
        .code("pt"), .code("nl"), .code("ru"), .code("uk"), .code("pl"),
        .code("tr"), .code("ar"), .code("fa"), .code("hi"), .code("zh"),
        .code("ja"), .code("ko"), .code("vi"), .code("id"), .code("sv"),
    ]
}
