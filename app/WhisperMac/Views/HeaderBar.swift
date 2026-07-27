import SwiftUI

/// Title bar replacement: wordmark, live status, engine chip, options menu.
struct HeaderBar: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            wordmark

            Divider().frame(height: 22)

            status

            Spacer(minLength: 12)

            enginePill
            optionsMenu
        }
        .padding(.leading, Metrics.titleBarInset)
        .padding(.trailing, Metrics.gutter)
        .padding(.vertical, 14)
        .background(Palette.canvasEdge.opacity(0.6))
    }

    // MARK: Pieces

    private var wordmark: some View {
        HStack(spacing: 9) {
            Image(systemName: "waveform.circle.fill")
                .font(.system(size: 19, weight: .medium))
                .foregroundStyle(Palette.accent)
            VStack(alignment: .leading, spacing: -1) {
                Text("WhisperMac")
                    .font(Type.title)
                    .tracking(-0.2)
                    .foregroundStyle(Palette.textPrimary)
                Text("local transcription")
                    .font(Type.eyebrow)
                    .tracking(0.7)
                    .textCase(.uppercase)
                    .foregroundStyle(Palette.textTertiary)
            }
        }
        .help("Everything runs on this Mac — nothing is uploaded.")
    }

    private var status: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(statusTint)
                .frame(width: 7, height: 7)
                .overlay(
                    Circle()
                        .stroke(statusTint.opacity(0.35), lineWidth: 5)
                        .scaleEffect(model.hasWork ? 1.6 : 1)
                        .opacity(model.hasWork ? 0.9 : 0)
                        .animation(
                            model.hasWork
                                ? .easeInOut(duration: 1.1).repeatForever(autoreverses: true)
                                : Metrics.fastAnimation,
                            value: model.hasWork
                        )
                )
            Text(model.headlineStatus)
                .font(Type.captionMedium)
                .foregroundStyle(Palette.textSecondary)
                .lineLimit(1)
        }
    }

    private var statusTint: Color {
        switch model.connection {
        case .offline: Palette.danger
        case .connecting: Palette.textTertiary
        case .online: model.hasWork ? Palette.accent : Palette.success
        }
    }

    private var enginePill: some View {
        HStack(spacing: 6) {
            Image(systemName: "cpu")
                .font(.system(size: 10, weight: .semibold))
            Text(model.modelLabel)
                .font(Type.monoSmall)
            if let health = model.health, health.modelLoaded {
                Image(systemName: "bolt.fill")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundStyle(Palette.success)
            }
        }
        .foregroundStyle(Palette.textSecondary)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Capsule().fill(Palette.surface))
        .overlay(Capsule().strokeBorder(Palette.hairline, lineWidth: 1))
        .help(engineTooltip)
    }

    private var engineTooltip: String {
        guard let health = model.health else { return "Engine offline" }
        var lines = ["Model: \(health.model)"]
        if let device = health.engine.device { lines.append("Device: \(device)") }
        if let version = health.engine.mlxVersion { lines.append("MLX \(version)") }
        lines.append(health.modelLoaded ? "Weights loaded" : "Weights not loaded yet")
        if let ffmpeg = health.ffmpeg { lines.append(ffmpeg) }
        return lines.joined(separator: "\n")
    }

    private var optionsMenu: some View {
        Menu {
            Section("On drop") {
                Toggle("Start transcribing immediately", isOn: $model.autoStart)
                Toggle("Reveal transcript in Finder when done", isOn: $model.revealWhenDone)
            }
            Section("Language") {
                Picker("Language", selection: $model.language) {
                    ForEach(LanguageOption.common) { option in
                        Text(option.label).tag(option)
                    }
                }
                .pickerStyle(.inline)
                Toggle("Translate to English", isOn: $model.translateToEnglish)
            }
            Divider()
            Button("Open Output Folder") { model.openOutputFolder() }
            Button("Choose Output Folder…") { model.chooseOutputFolder() }
            if !model.connection.isOnline, model.canStartEngine {
                Divider()
                Button("Start Engine") { model.startEngine() }
            }
        } label: {
            Image(systemName: "slider.horizontal.3")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Palette.textSecondary)
                .frame(width: 30, height: 26)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .frame(width: 30)
        .background(Capsule().fill(Palette.surface))
        .overlay(Capsule().strokeBorder(Palette.hairline, lineWidth: 1))
        .help("Options")
    }
}
