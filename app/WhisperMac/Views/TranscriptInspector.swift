import SwiftUI

/// Right-hand panel: the finished transcript, searchable and timestamped.
///
/// This is also the seam the roadmap features hang off — segment rows already
/// carry speaker and word data, so diarization labels and inline subtitle
/// editing land here without a redesign.
struct TranscriptInspector: View {
    @EnvironmentObject private var model: AppModel
    @State private var query = ""

    private var segments: [SegmentDTO] {
        guard let all = model.transcript?.segments else { return [] }
        let needle = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard !needle.isEmpty else { return all }
        return all.filter { $0.text.lowercased().contains(needle) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Hairline()

            if let item = model.selectedItem, item.stage != .completed {
                placeholder(
                    symbol: item.stage.symbol,
                    title: item.stage == .failed ? "No transcript" : "Transcript pending",
                    message: item.stage == .failed
                        ? (item.errorMessage ?? "This file failed to transcribe.")
                        : "It will appear here the moment this file finishes."
                )
            } else if model.transcript == nil {
                placeholder(
                    symbol: "text.alignleft",
                    title: model.isLoadingTranscript ? "Loading transcript…" : "No transcript selected",
                    message: "Pick a finished file in the queue to read, search and copy it."
                )
            } else {
                searchField
                segmentList
                footer
            }
        }
        .frame(width: Metrics.inspectorWidth)
        .background(Palette.canvasEdge.opacity(0.5))
    }

    // MARK: Pieces

    private var header: some View {
        HStack(spacing: 8) {
            Text("Transcript")
                .eyebrowStyle()
            Spacer()
            if let transcript = model.transcript {
                Text("\(transcript.wordCount) words")
                    .font(Type.monoSmall)
                    .foregroundStyle(Palette.textTertiary)
                IconButton(symbol: "doc.on.doc", help: "Copy full transcript") {
                    model.copyTranscriptToClipboard()
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 13)
    }

    private var searchField: some View {
        HStack(spacing: 7) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Palette.textTertiary)
            TextField("Search transcript", text: $query)
                .textFieldStyle(.plain)
                .font(Type.caption)
            if !query.isEmpty {
                IconButton(symbol: "xmark.circle.fill", help: "Clear") { query = "" }
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Palette.surface)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(Palette.hairline, lineWidth: 1)
        )
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private var segmentList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 2) {
                if segments.isEmpty {
                    Text("No lines match “\(query)”.")
                        .font(Type.caption)
                        .foregroundStyle(Palette.textTertiary)
                        .padding(.horizontal, 14)
                        .padding(.top, 8)
                }
                ForEach(segments) { segment in
                    SegmentRow(segment: segment, highlight: query)
                }
            }
            .padding(.horizontal, 8)
            .padding(.bottom, 10)
        }
    }

    private var footer: some View {
        VStack(spacing: 0) {
            Hairline()
            HStack(spacing: 10) {
                if let language = Format.languageName(model.transcript?.language) {
                    MetaChip(symbol: "globe", text: language)
                }
                if let duration = model.transcript?.duration {
                    MetaChip(symbol: "clock", text: Format.clock(duration))
                }
                Spacer()
                if let item = model.selectedItem, let output = item.outputs.first {
                    GhostButton(title: "Reveal", symbol: "folder") {
                        model.reveal(output.url)
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 11)
        }
    }

    private func placeholder(symbol: String, title: String, message: String) -> some View {
        VStack(spacing: 9) {
            Spacer()
            Image(systemName: symbol)
                .font(.system(size: 22, weight: .light))
                .foregroundStyle(Palette.textTertiary)
            Text(title)
                .font(Type.captionMedium)
                .foregroundStyle(Palette.textSecondary)
            Text(message)
                .font(Type.caption)
                .foregroundStyle(Palette.textTertiary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
        }
        .padding(.horizontal, 28)
        .frame(maxWidth: .infinity)
    }
}

/// One timestamped line. Clicking copies just that line — the fastest way to
/// pull a quote out of a meeting.
private struct SegmentRow: View {
    let segment: SegmentDTO
    let highlight: String

    @State private var isHovering = false

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Text(Format.timecode(segment.start))
                    .font(Type.monoSmall)
                    .monospacedDigit()
                    .foregroundStyle(isHovering ? Palette.accent : Palette.textTertiary)
                if let speaker = segment.speaker {
                    Text(speaker)
                        .font(Type.eyebrow)
                        .tracking(0.5)
                        .foregroundStyle(Palette.info)
                }
                Spacer(minLength: 0)
                if isHovering {
                    Image(systemName: "doc.on.doc")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundStyle(Palette.textTertiary)
                }
            }
            Text(attributed)
                .font(Type.caption)
                .foregroundStyle(Palette.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(isHovering ? Palette.surface : Color.clear)
        )
        .onHover { isHovering = $0 }
        .onTapGesture {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(segment.text, forType: .string)
        }
        .help("Click to copy this line")
    }

    /// Bold the search term inside the line.
    private var attributed: AttributedString {
        var string = AttributedString(segment.text)
        let needle = highlight.trimmingCharacters(in: .whitespaces)
        guard !needle.isEmpty,
              let range = string.range(of: needle, options: .caseInsensitive) else {
            return string
        }
        string[range].inlinePresentationIntent = .stronglyEmphasized
        string[range].foregroundColor = Palette.accent
        return string
    }
}
