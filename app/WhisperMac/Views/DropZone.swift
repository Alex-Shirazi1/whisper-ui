import SwiftUI
import UniformTypeIdentifiers

/// The empty state and the drop target.
///
/// Deliberately large and off-centre rather than a dead-centre stock hero: the
/// waveform sits behind the copy so the panel reads as a piece of audio
/// hardware, and the dashed ring animates only while a drag is over it.
struct DropZone: View {
    @EnvironmentObject private var model: AppModel
    @Binding var isTargeted: Bool

    private var dashPhase: CGFloat { isTargeted ? 26 : 0 }

    var body: some View {
        ZStack {
            WaveformBackdrop()
                .opacity(isTargeted ? 0.9 : 0.55)
                .animation(Metrics.normalAnimation, value: isTargeted)

            VStack(alignment: .leading, spacing: 0) {
                Spacer(minLength: 0)

                HStack(alignment: .center, spacing: 26) {
                    glyph
                    copy
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 38)

                Spacer(minLength: 0)

                Hairline().padding(.horizontal, 38)

                HStack(spacing: 14) {
                    Text("Supported")
                        .eyebrowStyle()
                    Text("MP4 · MOV · MKV · M4A · WAV · MP3 · FLAC · AAC")
                        .font(Type.monoSmall)
                        .foregroundStyle(Palette.textTertiary)
                    Spacer()
                    Text("Everything stays on this Mac")
                        .font(Type.caption)
                        .foregroundStyle(Palette.textTertiary)
                }
                .padding(.horizontal, 38)
                .padding(.vertical, 16)
            }
        }
        .frame(minHeight: 320)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(isTargeted ? Palette.accentSoft : Palette.surface)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(
                    isTargeted ? Palette.accent : Palette.hairlineStrong,
                    style: StrokeStyle(
                        lineWidth: isTargeted ? 2 : 1.5,
                        dash: [7, 6],
                        dashPhase: dashPhase
                    )
                )
        )
        .animation(
            isTargeted
                ? .linear(duration: 0.9).repeatForever(autoreverses: false)
                : Metrics.fastAnimation,
            value: dashPhase
        )
        .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private var glyph: some View {
        ZStack {
            Circle()
                .fill(isTargeted ? Palette.accent : Palette.surfaceSunken)
                .frame(width: 78, height: 78)
            Image(systemName: isTargeted ? "arrow.down.circle.fill" : "waveform.badge.plus")
                .font(.system(size: 30, weight: .light))
                .foregroundStyle(isTargeted ? .white : Palette.textSecondary)
        }
        .scaleEffect(isTargeted ? 1.06 : 1)
        .animation(Metrics.springAnimation, value: isTargeted)
    }

    private var copy: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(isTargeted ? "Release to transcribe" : "Drop a video or audio file")
                .font(Type.display)
                .tracking(-0.4)
                .foregroundStyle(Palette.textPrimary)

            Text(model.autoStart
                 ? "Transcription starts as soon as you drop. Everything runs locally on the Neural Engine and GPU."
                 : "Files queue up here — press Transcribe when you're ready.")
                .font(Type.body)
                .foregroundStyle(Palette.textSecondary)
                .frame(maxWidth: 430, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 10) {
                PrimaryButton(title: "Choose Files…", symbol: "folder") {
                    model.chooseFiles()
                }
                GhostButton(title: "Open Output Folder", symbol: "arrow.up.forward.app") {
                    model.openOutputFolder()
                }
            }
            .padding(.top, 4)
        }
    }
}

/// Decorative, deterministic waveform. No assets, no randomness — the same
/// shape every launch, which keeps screenshots stable.
private struct WaveformBackdrop: View {
    var body: some View {
        Canvas { context, size in
            let barCount = 110
            let spacing: CGFloat = 2
            let barWidth = max(1.5, (size.width - CGFloat(barCount) * spacing) / CGFloat(barCount))
            let midY = size.height / 2

            for index in 0..<barCount {
                let t = Double(index) / Double(barCount)
                // Layered sines read as speech without looking synthetic.
                let envelope = 0.35 + 0.65 * pow(sin(t * .pi), 0.7)
                let detail =
                    sin(t * 41.0) * 0.45
                    + sin(t * 17.0 + 1.7) * 0.32
                    + sin(t * 83.0 + 0.4) * 0.23
                let amplitude = abs(detail) * envelope
                let barHeight = max(2, CGFloat(amplitude) * size.height * 0.62)
                let x = CGFloat(index) * (barWidth + spacing)

                let bar = Path(
                    roundedRect: CGRect(
                        x: x, y: midY - barHeight / 2, width: barWidth, height: barHeight
                    ),
                    cornerRadius: barWidth / 2
                )
                context.fill(bar, with: .color(Palette.waveform))
            }
        }
        .padding(.horizontal, 30)
        .allowsHitTesting(false)
    }
}

// MARK: - Drop handling

extension View {
    /// Shared file-drop wiring.
    ///
    /// `loadDataRepresentation` for `public.file-url` is the reliable path for
    /// Finder drags; `loadObject(ofClass: URL.self)` silently misses some
    /// sources.
    func fileDropTarget(
        isTargeted: Binding<Bool>,
        onDrop: @escaping ([URL]) -> Void
    ) -> some View {
        self.onDrop(of: [.fileURL], isTargeted: isTargeted) { providers in
            let identifier = UTType.fileURL.identifier
            for provider in providers where provider.hasItemConformingToTypeIdentifier(identifier) {
                provider.loadDataRepresentation(forTypeIdentifier: identifier) { data, _ in
                    guard let data,
                          let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
                    Task { @MainActor in onDrop([url]) }
                }
            }
            return !providers.isEmpty
        }
    }
}
