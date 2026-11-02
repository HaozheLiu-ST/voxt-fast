// LaunchPresentationPolicy.swift
// Provides Launch Presentation Policy for app lifecycle and routing.

import AppKit
import Carbon

enum LaunchPresentationPolicy {
    @MainActor
    static func shouldPresentMainWindowOnLaunch(defaults: UserDefaults = .standard) -> Bool {
        shouldPresentMainWindowOnLaunch(
            isLaunchedAsLoginItem: LaunchSourceDetector.isLaunchedAsLoginItem,
            showMainWindowOnLaunch: defaults.bool(forKey: AppPreferenceKey.showMainWindowOnLaunch)
        )
    }

    /// A login-item launch is always silent. A manual launch opens the main window only when
    /// "Show main window on launch" is on (off by default); otherwise Voxt starts in the menu
    /// bar, and the window opens from the status menu or by opening the app again. The
    /// first-run onboarding window is presented separately and is not affected.
    nonisolated static func shouldPresentMainWindowOnLaunch(
        isLaunchedAsLoginItem: Bool,
        showMainWindowOnLaunch: Bool
    ) -> Bool {
        !isLaunchedAsLoginItem && showMainWindowOnLaunch
    }
}

enum LaunchSourceDetector {
    @MainActor
    static var isLaunchedAsLoginItem: Bool {
        guard let appleEvent = NSAppleEventManager.shared().currentAppleEvent else {
            return false
        }

        return appleEvent.paramDescriptor(forKeyword: AEKeyword(keyAELaunchedAsLogInItem)) != nil
    }
}
