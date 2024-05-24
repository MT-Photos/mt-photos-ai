//
//  chinese_clipApp.swift
//  chinese_clip
//
//  Created by Yixuan Si on 5/22/24.
//

//import SwiftUI
//
//@main
//struct chinese_clipApp: App {
//    var body: some Scene {
//        WindowGroup {
//            ContentView()
//        }
//    }
//}

import SwiftUI

@main
struct chinese_clipApp: App {
    @StateObject var viewModel = WebSocketViewModel()  // Shared ViewModel for the app

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)  // Provide the ViewModel to the views
                .onOpenURL { url in
                    handleURL(url)
                }
        }
    }

    private func handleURL(_ url: URL) {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: true),
              let queryItems = components.queryItems else { return }

        if let serverAddress = queryItems.first(where: { $0.name == "serverAddress" })?.value,
           let key = queryItems.first(where: { $0.name == "key" })?.value {
            // Update the ViewModel properties
            DispatchQueue.main.async {
                viewModel.serverAddress = serverAddress
                viewModel.clientKey = key
            }
        }
    }
}
