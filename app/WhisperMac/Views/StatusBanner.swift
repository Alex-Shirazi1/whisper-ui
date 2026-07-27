import SwiftUI

/// Inline banner for conditions the user can act on: engine offline, ffmpeg
/// missing, or a transient notice. Never a modal — a dialog for "the engine is
/// starting" would be worse than the problem.
struct StatusBanner: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 8) {
            if case .offline(let detail) = model.connection {
                banner(
                    tone: .error,
                    symbol: "bolt.horizontal.circle",
                    title: model.isStartingEngine ? "Starting the transcription engine…" : "Engine offline",
                    detail: model.isStartingEngine
                        ? "Loading Python and MLX. This takes a few seconds the first time."
                        : detail,
                    action: model.canStartEngine && !model.isStartingEngine
                        ? .init(title: "Start Engine", handler: model.startEngine)
                        : nil
                )
            }

            if let health = model.health, health.ffmpeg == nil {
                banner(
                    tone: .warning,
                    symbol: "exclamationmark.triangle",
                    title: "ffmpeg is not installed",
                    detail: "WhisperMac needs it to read audio out of media files. Install it with: brew install ffmpeg",
                    action: nil
                )
            }

            if let health = model.health, !health.engine.available {
                banner(
                    tone: .error,
                    symbol: "cpu",
                    title: "MLX engine unavailable",
                    detail: health.engine.error ?? "Run ./scripts/install.sh to repair the Python environment.",
                    action: nil
                )
            }

            if let notice = model.notice {
                banner(
                    tone: .info,
                    symbol: "info.circle",
                    title: notice,
                    detail: nil,
                    action: .init(title: "Dismiss") { model.notice = nil }
                )
            }
        }
        .animation(Metrics.normalAnimation, value: model.connection)
        .animation(Metrics.normalAnimation, value: model.notice)
    }

    // MARK: Builder

    private enum Tone {
        case error, warning, info

        var tint: Color {
            switch self {
            case .error: Palette.danger
            case .warning: Palette.accent
            case .info: Palette.info
            }
        }

        var wash: Color {
            switch self {
            case .error: Palette.dangerSoft
            case .warning: Palette.accentSoft
            case .info: Palette.surfaceSunken
            }
        }
    }

    private struct BannerAction {
        let title: String
        let handler: () -> Void
    }

    private func banner(
        tone: Tone,
        symbol: String,
        title: String,
        detail: String?,
        action: BannerAction?
    ) -> some View {
        HStack(alignment: .top, spacing: 11) {
            Image(systemName: symbol)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(tone.tint)
                .padding(.top, 1)

            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(Type.captionMedium)
                    .foregroundStyle(Palette.textPrimary)
                if let detail {
                    Text(detail)
                        .font(Type.caption)
                        .foregroundStyle(Palette.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            }

            Spacer(minLength: 8)

            if let action {
                GhostButton(title: action.title, action: action.handler)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .background(
            RoundedRectangle(cornerRadius: 11, style: .continuous)
                .fill(tone.wash)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 11, style: .continuous)
                .strokeBorder(tone.tint.opacity(0.28), lineWidth: 1)
        )
        .transition(.opacity.combined(with: .move(edge: .top)))
    }
}
