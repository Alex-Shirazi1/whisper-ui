import AppKit
import SwiftUI

// MARK: - Palette
//
// Visual direction: "studio console" — a warm paper canvas in light mode, a
// graphite one in dark, layered surfaces with hairline borders, and a single
// burnt-amber accent reserved for the one action that matters (transcribe).
// Every colour is a dynamic NSColor so both appearances are deliberate rather
// than one being an inverted afterthought.

enum Palette {
    static let canvas = dynamic(light: 0xF4F2EF, dark: 0x121214)
    static let canvasEdge = dynamic(light: 0xE8E4DE, dark: 0x0C0C0E)
    static let surface = dynamic(light: 0xFFFFFF, dark: 0x1B1B20)
    static let surfaceRaised = dynamic(light: 0xFFFFFF, dark: 0x24242B)
    static let surfaceSunken = dynamic(light: 0xEEEBE6, dark: 0x0F0F12)

    static let hairline = dynamic(light: 0x000000, lightAlpha: 0.10, dark: 0xFFFFFF, darkAlpha: 0.10)
    static let hairlineStrong = dynamic(light: 0x000000, lightAlpha: 0.18, dark: 0xFFFFFF, darkAlpha: 0.18)

    static let textPrimary = dynamic(light: 0x15151A, dark: 0xF3F1EE)
    static let textSecondary = dynamic(light: 0x15151A, lightAlpha: 0.58, dark: 0xF3F1EE, darkAlpha: 0.58)
    static let textTertiary = dynamic(light: 0x15151A, lightAlpha: 0.36, dark: 0xF3F1EE, darkAlpha: 0.38)

    static let accent = dynamic(light: 0xD1490F, dark: 0xFF7A45)
    static let accentSoft = dynamic(light: 0xD1490F, lightAlpha: 0.10, dark: 0xFF7A45, darkAlpha: 0.16)
    static let accentTrail = dynamic(light: 0xE8952A, dark: 0xFFC46B)

    static let success = dynamic(light: 0x1F7A52, dark: 0x54C892)
    static let successSoft = dynamic(light: 0x1F7A52, lightAlpha: 0.10, dark: 0x54C892, darkAlpha: 0.16)
    static let danger = dynamic(light: 0xB3301A, dark: 0xFF6E58)
    static let dangerSoft = dynamic(light: 0xB3301A, lightAlpha: 0.10, dark: 0xFF6E58, darkAlpha: 0.16)
    static let info = dynamic(light: 0x2F5FB3, dark: 0x7BA5F5)

    /// Waveform decoration in the drop zone.
    static let waveform = dynamic(light: 0x15151A, lightAlpha: 0.10, dark: 0xFFFFFF, darkAlpha: 0.12)

    static let progressGradient = LinearGradient(
        colors: [accent, accentTrail],
        startPoint: .leading,
        endPoint: .trailing
    )

    // MARK: Builders

    static func dynamic(
        light: UInt32,
        lightAlpha: Double = 1,
        dark: UInt32,
        darkAlpha: Double = 1
    ) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            return isDark
                ? NSColor(hex: dark, alpha: darkAlpha)
                : NSColor(hex: light, alpha: lightAlpha)
        })
    }
}

extension NSColor {
    convenience init(hex: UInt32, alpha: Double = 1) {
        self.init(
            srgbRed: CGFloat((hex >> 16) & 0xFF) / 255,
            green: CGFloat((hex >> 8) & 0xFF) / 255,
            blue: CGFloat(hex & 0xFF) / 255,
            alpha: CGFloat(alpha)
        )
    }
}

// MARK: - Typography
//
// Scale contrast does the hierarchy work: a 30pt display against a 10pt
// uppercase eyebrow, with monospaced digits everywhere a number changes so
// nothing twitches while a job runs.

enum Type {
    static let display = Font.system(size: 30, weight: .semibold, design: .default)
    static let title = Font.system(size: 17, weight: .semibold)
    static let headline = Font.system(size: 13.5, weight: .semibold)
    static let body = Font.system(size: 13, weight: .regular)
    static let bodyMedium = Font.system(size: 13, weight: .medium)
    static let caption = Font.system(size: 11.5, weight: .regular)
    static let captionMedium = Font.system(size: 11.5, weight: .medium)
    static let eyebrow = Font.system(size: 10, weight: .semibold)
    static let mono = Font.system(size: 11.5, weight: .medium, design: .monospaced)
    static let monoSmall = Font.system(size: 10.5, weight: .regular, design: .monospaced)
}

// MARK: - Metrics

enum Metrics {
    static let windowMinWidth: CGFloat = 940
    static let windowMinHeight: CGFloat = 660
    static let inspectorWidth: CGFloat = 340

    /// Clears the hidden-titlebar traffic lights.
    static let titleBarInset: CGFloat = 72

    static let gutter: CGFloat = 22
    static let stackGap: CGFloat = 14
    static let tightGap: CGFloat = 8

    static let cardRadius: CGFloat = 14
    static let controlRadius: CGFloat = 9
    static let pillRadius: CGFloat = 999

    static let fastAnimation = Animation.easeOut(duration: 0.16)
    static let normalAnimation = Animation.easeOut(duration: 0.28)
    static let springAnimation = Animation.spring(response: 0.38, dampingFraction: 0.82)
}

// MARK: - Reusable modifiers

struct CardSurface: ViewModifier {
    var raised: Bool = false
    var radius: CGFloat = Metrics.cardRadius

    func body(content: Content) -> some View {
        content
            .background(raised ? Palette.surfaceRaised : Palette.surface)
            .clipShape(RoundedRectangle(cornerRadius: radius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .strokeBorder(Palette.hairline, lineWidth: 1)
            )
            .shadow(color: .black.opacity(raised ? 0.14 : 0.06), radius: raised ? 18 : 8, y: raised ? 8 : 3)
    }
}

extension View {
    func cardSurface(raised: Bool = false, radius: CGFloat = Metrics.cardRadius) -> some View {
        modifier(CardSurface(raised: raised, radius: radius))
    }

    /// Uppercase section label used above every group of controls.
    func eyebrowStyle() -> some View {
        font(Type.eyebrow)
            .tracking(0.9)
            .textCase(.uppercase)
            .foregroundStyle(Palette.textTertiary)
    }
}
