// swift-tools-version:5.9
//
// WhisperMac is built with SwiftPM rather than an .xcodeproj so that
// `scripts/build.sh` can produce a signed .app bundle from a clean checkout
// with no Xcode project state to keep in sync. Open this package directly in
// Xcode (File > Open > app/Package.swift) if you prefer the IDE.

import PackageDescription

let package = Package(
    name: "WhisperMac",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "WhisperMac", targets: ["WhisperMac"])
    ],
    targets: [
        .executableTarget(
            name: "WhisperMac",
            path: "WhisperMac",
            // Both are consumed by scripts/build.sh when assembling the bundle,
            // not compiled as sources.
            exclude: ["Info.plist", "Assets.xcassets"]
        )
    ]
)
