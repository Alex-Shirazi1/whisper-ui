import SwiftUI

/// The scrolling list of queued files, with a compact toolbar above it.
struct QueueView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            toolbar

            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(model.items) { item in
                        JobRow(item: item, isSelected: model.selection == item.id)
                            .transition(.opacity.combined(with: .move(edge: .top)))
                    }
                }
                .padding(.bottom, 4)
            }
            .scrollContentBackground(.hidden)
        }
        .animation(Metrics.normalAnimation, value: model.items.count)
    }

    private var toolbar: some View {
        HStack(spacing: 10) {
            Text("Queue")
                .eyebrowStyle()
            Text("\(model.items.count)")
                .font(Type.monoSmall)
                .foregroundStyle(Palette.textTertiary)

            Spacer()

            if !model.activeItems.isEmpty {
                GhostButton(title: "Cancel All", symbol: "stop.circle", isDestructive: true) {
                    model.cancelAll()
                }
            }
            if !model.finishedItems.isEmpty {
                GhostButton(title: "Clear Finished", symbol: "checkmark.circle") {
                    model.clearFinished()
                }
            }
            GhostButton(title: "Add Files…", symbol: "plus") {
                model.chooseFiles()
            }
        }
    }
}
