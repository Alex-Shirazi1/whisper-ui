import Foundation

/// Persisted user choices.
///
/// Everything here is a convenience, never a correctness dependency — a fresh
/// install (or a wiped defaults domain) falls back to values that work.
enum Preferences {
    private enum Key {
        static let formats = "formats"
        static let outputFolder = "outputFolder"
        static let language = "language"
        static let autoStart = "autoStart"
        static let revealWhenDone = "revealWhenDone"
        static let translate = "translate"
    }

    private static let defaults = UserDefaults.standard

    // MARK: Formats

    static var formats: Set<OutputFormat> {
        get {
            guard let raw = defaults.string(forKey: Key.formats) else { return [.txt, .srt] }
            let parsed = Set(raw.split(separator: ",").compactMap { OutputFormat(rawValue: String($0)) })
            // Never leave the user with nothing selected.
            return parsed.isEmpty ? [.txt] : parsed
        }
        set {
            let value = newValue.isEmpty ? [OutputFormat.txt] : newValue.ordered
            defaults.set(value.map(\.rawValue).joined(separator: ","), forKey: Key.formats)
        }
    }

    // MARK: Output folder

    static var outputFolder: URL? {
        get {
            guard let path = defaults.string(forKey: Key.outputFolder), !path.isEmpty else {
                return nil
            }
            let url = URL(fileURLWithPath: path)
            // A folder on an ejected volume should not brick the UI.
            return FileManager.default.fileExists(atPath: url.path) ? url : nil
        }
        set { defaults.set(newValue?.path, forKey: Key.outputFolder) }
    }

    // MARK: Language / task

    static var language: LanguageOption {
        get {
            guard let raw = defaults.string(forKey: Key.language), raw != "auto", !raw.isEmpty else {
                return .auto
            }
            return .code(raw)
        }
        set { defaults.set(newValue.id, forKey: Key.language) }
    }

    static var translateToEnglish: Bool {
        get { defaults.bool(forKey: Key.translate) }
        set { defaults.set(newValue, forKey: Key.translate) }
    }

    // MARK: Behaviour

    static var autoStart: Bool {
        get { defaults.object(forKey: Key.autoStart) as? Bool ?? true }
        set { defaults.set(newValue, forKey: Key.autoStart) }
    }

    static var revealWhenDone: Bool {
        get { defaults.object(forKey: Key.revealWhenDone) as? Bool ?? false }
        set { defaults.set(newValue, forKey: Key.revealWhenDone) }
    }
}
