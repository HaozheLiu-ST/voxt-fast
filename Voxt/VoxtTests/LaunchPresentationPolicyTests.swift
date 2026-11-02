// LaunchPresentationPolicyTests.swift
// Covers when a launch opens the main window.

import XCTest
@testable import Voxt

@MainActor
final class LaunchPresentationPolicyTests: XCTestCase {
    func testManualLaunchOpensTheMainWindowOnlyWhenThePreferenceIsOn() {
        XCTAssertFalse(LaunchPresentationPolicy.shouldPresentMainWindowOnLaunch(
            isLaunchedAsLoginItem: false,
            showMainWindowOnLaunch: false
        ))
        XCTAssertTrue(LaunchPresentationPolicy.shouldPresentMainWindowOnLaunch(
            isLaunchedAsLoginItem: false,
            showMainWindowOnLaunch: true
        ))
    }

    func testLoginItemLaunchNeverOpensTheMainWindow() {
        for showMainWindowOnLaunch in [false, true] {
            XCTAssertFalse(LaunchPresentationPolicy.shouldPresentMainWindowOnLaunch(
                isLaunchedAsLoginItem: true,
                showMainWindowOnLaunch: showMainWindowOnLaunch
            ))
        }
    }

    func testThePreferenceIsOffUnlessStored() throws {
        let suiteName = "LaunchPresentationPolicyTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        // Unit tests are not launched as a login item.
        XCTAssertFalse(LaunchPresentationPolicy.shouldPresentMainWindowOnLaunch(defaults: defaults))
        defaults.set(true, forKey: AppPreferenceKey.showMainWindowOnLaunch)
        XCTAssertTrue(LaunchPresentationPolicy.shouldPresentMainWindowOnLaunch(defaults: defaults))
    }
}
