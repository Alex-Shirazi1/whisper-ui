#!/usr/bin/env swift
//
// Renders WhisperMac.iconset with Core Graphics — no design tools, no binary
// assets in the repo. build.sh runs `iconutil` over the result.
//
//   swift scripts/make_icon.swift <output.iconset>
//
import AppKit
import Foundation

let arguments = CommandLine.arguments
guard arguments.count == 2 else {
    FileHandle.standardError.write(Data("usage: make_icon.swift <output.iconset>\n".utf8))
    exit(2)
}

let outputURL = URL(fileURLWithPath: arguments[1])
try? FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)

/// Deterministic bar heights — the same silhouette as the in-app waveform.
let bars: [CGFloat] = [0.30, 0.52, 0.74, 0.96, 0.66, 0.42, 0.86, 0.58, 0.34, 0.70, 0.46, 0.26]

func drawIcon(side: CGFloat, into context: CGContext) {
    let rect = CGRect(x: 0, y: 0, width: side, height: side)

    // macOS icon grid: the tile is inset, with a squircle silhouette.
    let inset = side * 0.085
    let tile = rect.insetBy(dx: inset, dy: inset)
    let radius = tile.width * 0.235
    let squircle = CGPath(roundedRect: tile, cornerWidth: radius, cornerHeight: radius, transform: nil)

    context.saveGState()
    context.addPath(squircle)
    context.clip()

    // Warm amber gradient, deepest at the bottom-right.
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let gradient = CGGradient(
        colorsSpace: colorSpace,
        colors: [
            CGColor(srgbRed: 1.00, green: 0.62, blue: 0.24, alpha: 1),
            CGColor(srgbRed: 0.82, green: 0.26, blue: 0.06, alpha: 1),
        ] as CFArray,
        locations: [0, 1]
    )!
    context.drawLinearGradient(
        gradient,
        start: CGPoint(x: tile.minX, y: tile.maxY),
        end: CGPoint(x: tile.maxX, y: tile.minY),
        options: []
    )

    // Soft top-light sheen for depth.
    let sheen = CGGradient(
        colorsSpace: colorSpace,
        colors: [
            CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.28),
            CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.0),
        ] as CFArray,
        locations: [0, 1]
    )!
    context.drawLinearGradient(
        sheen,
        start: CGPoint(x: tile.midX, y: tile.maxY),
        end: CGPoint(x: tile.midX, y: tile.midY),
        options: []
    )
    context.restoreGState()

    // Waveform.
    let plotWidth = tile.width * 0.62
    let barWidth = plotWidth / CGFloat(bars.count) * 0.52
    let spacing = (plotWidth - barWidth * CGFloat(bars.count)) / CGFloat(bars.count - 1)
    let startX = tile.midX - plotWidth / 2
    let maxHeight = tile.height * 0.46

    context.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.97))
    for (index, amplitude) in bars.enumerated() {
        let height = max(barWidth, maxHeight * amplitude)
        let x = startX + CGFloat(index) * (barWidth + spacing)
        let bar = CGRect(x: x, y: tile.midY - height / 2, width: barWidth, height: height)
        context.addPath(
            CGPath(
                roundedRect: bar,
                cornerWidth: barWidth / 2,
                cornerHeight: barWidth / 2,
                transform: nil
            )
        )
        context.fillPath()
    }
}

func write(side: Int, scale: Int, filename: String) throws {
    let pixels = side * scale
    guard let representation = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: pixels,
        pixelsHigh: pixels,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else {
        throw NSError(domain: "make_icon", code: 1)
    }

    NSGraphicsContext.saveGraphicsState()
    let graphicsContext = NSGraphicsContext(bitmapImageRep: representation)!
    NSGraphicsContext.current = graphicsContext
    drawIcon(side: CGFloat(pixels), into: graphicsContext.cgContext)
    NSGraphicsContext.restoreGraphicsState()

    guard let data = representation.representation(using: .png, properties: [:]) else {
        throw NSError(domain: "make_icon", code: 2)
    }
    try data.write(to: outputURL.appendingPathComponent(filename))
}

// The exact set `iconutil` expects.
let variants: [(Int, Int, String)] = [
    (16, 1, "icon_16x16.png"),
    (16, 2, "icon_16x16@2x.png"),
    (32, 1, "icon_32x32.png"),
    (32, 2, "icon_32x32@2x.png"),
    (128, 1, "icon_128x128.png"),
    (128, 2, "icon_128x128@2x.png"),
    (256, 1, "icon_256x256.png"),
    (256, 2, "icon_256x256@2x.png"),
    (512, 1, "icon_512x512.png"),
    (512, 2, "icon_512x512@2x.png"),
]

for (side, scale, filename) in variants {
    try write(side: side, scale: scale, filename: filename)
}
print("Wrote \(variants.count) icon variants to \(outputURL.path)")
