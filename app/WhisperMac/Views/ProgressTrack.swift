import SwiftUI

/// The progress bar.
///
/// Determinate mode fills a gradient; indeterminate mode sweeps a highlight for
/// the stages with no measurable position (downloading and loading weights).
/// Both animate `offset`/`opacity` only, so the work stays on the compositor.
struct ProgressTrack: View {
    var value: Double
    var isIndeterminate: Bool = false
    var height: CGFloat = 6
    var tint: LinearGradient = Palette.progressGradient

    @State private var sweep: CGFloat = -0.4

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let filled = max(0, min(1, value)) * width

            ZStack(alignment: .leading) {
                Capsule(style: .continuous)
                    .fill(Palette.surfaceSunken)

                if isIndeterminate {
                    Capsule(style: .continuous)
                        .fill(tint)
                        .frame(width: max(width * 0.32, 48))
                        .offset(x: sweep * width)
                        .mask(Capsule(style: .continuous))
                } else {
                    Capsule(style: .continuous)
                        .fill(tint)
                        .frame(width: filled)
                        .animation(Metrics.normalAnimation, value: filled)
                }
            }
            .frame(height: height)
            .clipShape(Capsule(style: .continuous))
        }
        .frame(height: height)
        .onAppear(perform: startSweepIfNeeded)
        .onChange(of: isIndeterminate) { _, _ in startSweepIfNeeded() }
    }

    private func startSweepIfNeeded() {
        guard isIndeterminate else {
            sweep = -0.4
            return
        }
        withAnimation(.easeInOut(duration: 1.15).repeatForever(autoreverses: false)) {
            sweep = 1.05
        }
    }
}

/// Thin window-level bar shown under the header while anything is running.
struct AggregateProgressBar: View {
    let progress: Double
    let isActive: Bool

    var body: some View {
        ProgressTrack(value: progress, height: 2.5)
            .opacity(isActive ? 1 : 0)
            .animation(Metrics.normalAnimation, value: isActive)
    }
}
