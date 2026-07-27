import SwiftUI

/// Bottom bar: output formats, destination folder, and the primary action.
struct ControlBar: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            Hairline()

            HStack(alignment: .center, spacing: 22) {
                formatPicker
                Divider().frame(height: 34)
                destination
                Spacer(minLength: 12)
                actions
            }
            .padding(.horizontal, Metrics.gutter)
            .padding(.vertical, 14)
        }
        .background(Palette.canvasEdge.opacity(0.75))
    }

    // MARK: Formats

    private var formatPicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Output")
                .eyebrowStyle()
            HStack(spacing: 6) {
                ForEach(OutputFormat.allCases) { format in
                    FormatToggle(
                        format: format,
                        isOn: model.formats.contains(format),
                        action: { model.toggle(format) }
                    )
                }
            }
        }
    }

    // MARK: Destination

    private var destination: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Save to")
                .eyebrowStyle()
            HStack(spacing: 6) {
                Button {
                    model.chooseOutputFolder()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "folder")
                            .font(.system(size: 11, weight: .medium))
                        Text(Format.shortPath(model.outputFolder))
                            .font(Type.monoSmall)
                            .lineLimit(1)
                            .truncationMode(.head)
                    }
                    .foregroundStyle(Palette.textSecondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(
                        RoundedRectangle(cornerRadius: 7, style: .continuous)
                            .fill(Palette.surface)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 7, style: .continuous)
                            .strokeBorder(Palette.hairline, lineWidth: 1)
                    )
                }
                .buttonStyle(.plain)
                .help("Choose where transcripts are saved")
                .frame(maxWidth: 260)

                IconButton(symbol: "arrow.up.forward.app", help: "Open output folder") {
                    model.openOutputFolder()
                }
            }
        }
    }

    // MARK: Actions

    private var actions: some View {
        HStack(spacing: 10) {
            if model.hasWork {
                GhostButton(title: "Cancel", symbol: "stop.fill", isDestructive: true) {
                    model.cancelAll()
                }
            }
            PrimaryButton(
                title: primaryTitle,
                symbol: model.hasWork ? "waveform" : "text.viewfinder",
                isEnabled: model.canStart || (!model.connection.isOnline && model.canStartEngine)
            ) {
                if model.connection.isOnline {
                    model.startAll()
                } else {
                    model.startEngine()
                }
            }
        }
    }

    private var primaryTitle: String {
        if !model.connection.isOnline { return model.isStartingEngine ? "Starting…" : "Start Engine" }
        let waiting = model.waitingItems.count
        if waiting > 1 { return "Transcribe \(waiting) Files" }
        if waiting == 1 { return "Transcribe" }
        return model.hasWork ? "Transcribing…" : "Transcribe"
    }
}

/// A single format toggle. Selected state uses the accent, not a checkbox — the
/// row of four reads as one control.
private struct FormatToggle: View {
    let format: OutputFormat
    let isOn: Bool
    let action: () -> Void

    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: format.symbol)
                    .font(.system(size: 10, weight: .semibold))
                Text(format.label)
                    .font(Type.captionMedium)
                    .tracking(0.3)
            }
            .foregroundStyle(isOn ? .white : Palette.textSecondary)
            .padding(.horizontal, 11)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(isOn ? Palette.accent : (isHovering ? Palette.surfaceSunken : Palette.surface))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .strokeBorder(isOn ? Color.clear : Palette.hairline, lineWidth: 1)
            )
            .shadow(color: Palette.accent.opacity(isOn ? 0.25 : 0), radius: 5, y: 2)
        }
        .buttonStyle(.plain)
        .onHover { isHovering = $0 }
        .help("\(format.label) — \(format.blurb)")
        .animation(Metrics.fastAnimation, value: isOn)
        .animation(Metrics.fastAnimation, value: isHovering)
    }
}
