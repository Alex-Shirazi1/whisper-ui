import SwiftUI

// MARK: - Buttons
//
// Written as views rather than ButtonStyles so hover, press and disabled states
// can all be designed rather than inherited from AppKit defaults.

struct PrimaryButton: View {
    let title: String
    var symbol: String?
    var tint: Color = Palette.accent
    var isEnabled: Bool = true
    let action: () -> Void

    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 7) {
                if let symbol {
                    Image(systemName: symbol).font(.system(size: 12, weight: .semibold))
                }
                Text(title).font(Type.headline)
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 18)
            .padding(.vertical, 9)
            .background(
                RoundedRectangle(cornerRadius: Metrics.controlRadius, style: .continuous)
                    .fill(tint)
                    .brightness(isHovering && isEnabled ? 0.06 : 0)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Metrics.controlRadius, style: .continuous)
                    .strokeBorder(.white.opacity(0.18), lineWidth: 1)
            )
            .shadow(color: tint.opacity(isEnabled ? 0.32 : 0), radius: isHovering ? 12 : 6, y: 3)
            .opacity(isEnabled ? 1 : 0.4)
            .scaleEffect(isHovering && isEnabled ? 1.015 : 1)
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .onHover { isHovering = $0 }
        .animation(Metrics.fastAnimation, value: isHovering)
    }
}

struct GhostButton: View {
    let title: String
    var symbol: String?
    var isEnabled: Bool = true
    var isDestructive: Bool = false
    let action: () -> Void

    @State private var isHovering = false

    private var foreground: Color {
        isDestructive ? Palette.danger : Palette.textPrimary
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                if let symbol {
                    Image(systemName: symbol).font(.system(size: 11.5, weight: .medium))
                }
                Text(title).font(Type.captionMedium)
            }
            .foregroundStyle(foreground.opacity(isEnabled ? 1 : 0.4))
            .padding(.horizontal, 13)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: Metrics.controlRadius, style: .continuous)
                    .fill(isHovering && isEnabled
                          ? (isDestructive ? Palette.dangerSoft : Palette.surfaceSunken)
                          : Color.clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Metrics.controlRadius, style: .continuous)
                    .strokeBorder(isHovering && isEnabled ? Palette.hairlineStrong : Palette.hairline,
                                  lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .onHover { isHovering = $0 }
        .animation(Metrics.fastAnimation, value: isHovering)
    }
}

/// Compact icon-only action used inside queue rows.
struct IconButton: View {
    let symbol: String
    let help: String
    var tint: Color = Palette.textSecondary
    let action: () -> Void

    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(isHovering ? tint : tint.opacity(0.7))
                .frame(width: 26, height: 26)
                .background(
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .fill(isHovering ? Palette.surfaceSunken : Color.clear)
                )
        }
        .buttonStyle(.plain)
        .help(help)
        .onHover { isHovering = $0 }
        .animation(Metrics.fastAnimation, value: isHovering)
    }
}

// MARK: - Indicators

/// Small stage badge: icon + label, tinted by outcome.
struct StageBadge: View {
    let stage: JobStage

    private var tint: Color {
        switch stage {
        case .completed: Palette.success
        case .failed: Palette.danger
        case .cancelled: Palette.textTertiary
        case .ready: Palette.textSecondary
        default: Palette.accent
        }
    }

    private var background: Color {
        switch stage {
        case .completed: Palette.successSoft
        case .failed: Palette.dangerSoft
        case .ready, .cancelled: Palette.surfaceSunken
        default: Palette.accentSoft
        }
    }

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: stage.symbol).font(.system(size: 9, weight: .bold))
            Text(stage.label).font(Type.eyebrow).tracking(0.5)
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 7)
        .padding(.vertical, 3.5)
        .background(Capsule().fill(background))
    }
}

/// Key/value chip used for durations, languages and speed.
struct MetaChip: View {
    let symbol: String
    let text: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: symbol).font(.system(size: 9.5, weight: .medium))
            Text(text).font(Type.monoSmall)
        }
        .foregroundStyle(Palette.textTertiary)
    }
}

/// Clickable chip for a written output file.
struct OutputChip: View {
    let output: OutputFileDTO
    let onOpen: () -> Void
    let onReveal: () -> Void

    @State private var isHovering = false

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: output.outputFormat?.symbol ?? "doc")
                .font(.system(size: 9.5, weight: .semibold))
            Text(output.format.uppercased()).font(Type.eyebrow).tracking(0.6)
        }
        .foregroundStyle(isHovering ? Palette.accent : Palette.textSecondary)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(isHovering ? Palette.accentSoft : Palette.surfaceSunken)
        )
        .onHover { isHovering = $0 }
        .onTapGesture(perform: onOpen)
        .contextMenu {
            Button("Open \(output.filename)", action: onOpen)
            Button("Show in Finder", action: onReveal)
        }
        .help("Open \(output.filename)")
        .animation(Metrics.fastAnimation, value: isHovering)
    }
}

struct Hairline: View {
    var body: some View {
        Rectangle()
            .fill(Palette.hairline)
            .frame(height: 1)
    }
}
