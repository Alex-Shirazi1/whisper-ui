import SwiftUI

/// One file in the queue.
///
/// Hierarchy inside the row: filename first, then the live status line, then the
/// bar, then metadata. Actions only appear on hover (or when the row is
/// selected) so a queue at rest stays quiet.
struct JobRow: View {
    let item: QueueItem
    let isSelected: Bool

    @EnvironmentObject private var model: AppModel
    @State private var isHovering = false

    private var showsBar: Bool {
        item.stage.isRunning || item.stage == .completed
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            header

            if showsBar {
                ProgressTrack(
                    value: item.stage == .completed ? 1 : item.progress,
                    isIndeterminate: item.job?.indeterminate ?? false,
                    tint: item.stage == .completed
                        ? LinearGradient(colors: [Palette.success, Palette.success],
                                         startPoint: .leading, endPoint: .trailing)
                        : Palette.progressGradient
                )
            }

            statusLine

            if let error = item.errorMessage, item.stage == .failed {
                Text(error)
                    .font(Type.caption)
                    .foregroundStyle(Palette.danger)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 1)
            }

            if !item.outputs.isEmpty {
                HStack(spacing: 6) {
                    ForEach(item.outputs, id: \.path) { output in
                        OutputChip(
                            output: output,
                            onOpen: { model.open(output.url) },
                            onReveal: { model.reveal(output.url) }
                        )
                    }
                }
                .padding(.top, 1)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .background(rowBackground)
        .overlay(alignment: .leading) {
            // Selection is marked with an accent rule rather than a fill, so the
            // row's own state colours stay readable.
            Rectangle()
                .fill(Palette.accent)
                .frame(width: 3)
                .opacity(isSelected ? 1 : 0)
        }
        .clipShape(RoundedRectangle(cornerRadius: Metrics.cardRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Metrics.cardRadius, style: .continuous)
                .strokeBorder(isSelected ? Palette.hairlineStrong : Palette.hairline, lineWidth: 1)
        )
        .contentShape(Rectangle())
        .onTapGesture { model.select(item) }
        .onHover { isHovering = $0 }
        .animation(Metrics.fastAnimation, value: isHovering)
        .contextMenu { contextMenuItems }
    }

    // MARK: Pieces

    private var rowBackground: some View {
        (isHovering ? Palette.surfaceRaised : Palette.surface)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Image(systemName: (item.job?.isVideo ?? isLikelyVideo) ? "film" : "waveform")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Palette.textTertiary)
                .frame(width: 14)

            Text(item.name)
                .font(Type.bodyMedium)
                .foregroundStyle(Palette.textPrimary)
                .lineLimit(1)
                .truncationMode(.middle)

            StageBadge(stage: item.stage)

            Spacer(minLength: 8)

            if item.stage.isRunning {
                Text(Format.percent(item.progress))
                    .font(Type.mono)
                    .monospacedDigit()
                    .foregroundStyle(Palette.textSecondary)
            }

            actions
                .opacity(isHovering || isSelected ? 1 : 0)
        }
    }

    private var statusLine: some View {
        HStack(spacing: 12) {
            Text(item.statusMessage)
                .font(Type.caption)
                .foregroundStyle(item.stage == .failed ? Palette.danger : Palette.textSecondary)
                .lineLimit(1)

            Spacer(minLength: 0)

            if let duration = item.job?.mediaDuration {
                MetaChip(symbol: "clock", text: Format.clock(duration))
            }
            if let language = Format.languageName(item.job?.detectedLanguage) {
                MetaChip(symbol: "globe", text: language)
            }
            if let speed = Format.speed(item.job?.speedFactor) {
                MetaChip(symbol: "bolt", text: speed)
            } else if let eta = Format.remaining(item.job?.etaSeconds), item.stage.isRunning {
                MetaChip(symbol: "hourglass", text: eta)
            }
        }
    }

    private var actions: some View {
        HStack(spacing: 2) {
            if item.canStart {
                IconButton(symbol: "play.fill", help: "Transcribe this file", tint: Palette.accent) {
                    Task { await model.submit(item) }
                }
            }
            if item.canCancel {
                IconButton(symbol: "stop.fill", help: "Cancel", tint: Palette.danger) {
                    model.cancel(item)
                }
            }
            if item.stage == .failed || item.stage == .cancelled {
                IconButton(symbol: "arrow.clockwise", help: "Try again") {
                    model.retry(item)
                }
            }
            if let first = item.outputs.first {
                IconButton(symbol: "folder", help: "Show in Finder") {
                    model.reveal(first.url)
                }
            }
            IconButton(symbol: "xmark", help: "Remove from queue") {
                model.remove(item)
            }
        }
    }

    @ViewBuilder
    private var contextMenuItems: some View {
        Button("Show Source in Finder") { model.reveal(item.url) }
        if item.canStart {
            Button("Transcribe Now") { Task { await model.submit(item) } }
        }
        if item.canCancel {
            Button("Cancel") { model.cancel(item) }
        }
        if item.stage == .failed || item.stage == .cancelled {
            Button("Try Again") { model.retry(item) }
        }
        Divider()
        ForEach(item.outputs, id: \.path) { output in
            Button("Open \(output.filename)") { model.open(output.url) }
        }
        if !item.outputs.isEmpty { Divider() }
        Button("Remove from Queue") { model.remove(item) }
    }

    private var isLikelyVideo: Bool {
        ["mp4", "mov", "m4v", "mkv", "webm", "avi"].contains(item.url.pathExtension.lowercased())
    }
}
