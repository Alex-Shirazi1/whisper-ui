import SwiftUI

/// Window composition: header, banners, main stage, inspector, control bar.
///
/// The whole window is a drop target, not just the empty state — once the queue
/// has rows, dropping anywhere still adds files.
struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @State private var isTargeted = false
    @State private var showsInspector = true

    var body: some View {
        VStack(spacing: 0) {
            HeaderBar()
            AggregateProgressBar(progress: model.overallProgress, isActive: model.hasWork)

            HStack(spacing: 0) {
                mainStage

                if showsInspector {
                    Divider()
                    TranscriptInspector()
                        .transition(.move(edge: .trailing).combined(with: .opacity))
                }
            }
            .frame(maxHeight: .infinity)

            ControlBar()
        }
        .background(Palette.canvas)
        .frame(
            minWidth: Metrics.windowMinWidth,
            minHeight: Metrics.windowMinHeight
        )
        .fileDropTarget(isTargeted: $isTargeted) { urls in
            model.add(urls: urls)
        }
        .animation(Metrics.normalAnimation, value: showsInspector)
        .onAppear { model.startMonitoring() }
        .onDisappear { model.stopMonitoring() }
        .toolbar { inspectorToggle }
        .focusedSceneValue(\.appModel, model)
    }

    // MARK: Main stage

    @ViewBuilder
    private var mainStage: some View {
        VStack(alignment: .leading, spacing: Metrics.stackGap) {
            StatusBanner()

            if model.items.isEmpty {
                DropZone(isTargeted: $isTargeted)
                    .frame(maxHeight: .infinity)
            } else {
                QueueView()
                dropHint
            }
        }
        .padding(.horizontal, Metrics.gutter)
        .padding(.top, Metrics.stackGap)
        .padding(.bottom, Metrics.stackGap)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .overlay(alignment: .center) {
            if isTargeted, !model.items.isEmpty {
                dropOverlay
            }
        }
    }

    private var dropHint: some View {
        HStack(spacing: 6) {
            Image(systemName: "arrow.down.doc")
                .font(.system(size: 10, weight: .medium))
            Text("Drop more files anywhere in this window")
                .font(Type.caption)
        }
        .foregroundStyle(Palette.textTertiary)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Palette.hairline, style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
        )
    }

    private var dropOverlay: some View {
        VStack(spacing: 10) {
            Image(systemName: "arrow.down.circle.fill")
                .font(.system(size: 34, weight: .light))
                .foregroundStyle(Palette.accent)
            Text("Add to queue")
                .font(Type.title)
                .foregroundStyle(Palette.textPrimary)
        }
        .padding(38)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(Palette.accent, lineWidth: 2)
        )
        .transition(.scale(scale: 0.96).combined(with: .opacity))
    }

    // MARK: Toolbar

    private var inspectorToggle: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            Button {
                showsInspector.toggle()
            } label: {
                Image(systemName: showsInspector ? "sidebar.right" : "sidebar.trailing")
            }
            .help(showsInspector ? "Hide transcript" : "Show transcript")
        }
    }
}

// MARK: - Focused value plumbing
//
// Lets the menu bar commands act on the window's model.

private struct AppModelFocusKey: FocusedValueKey {
    typealias Value = AppModel
}

extension FocusedValues {
    var appModel: AppModel? {
        get { self[AppModelFocusKey.self] }
        set { self[AppModelFocusKey.self] = newValue }
    }
}
