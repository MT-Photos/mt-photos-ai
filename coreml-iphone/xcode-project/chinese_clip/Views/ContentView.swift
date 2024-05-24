//import SwiftUI
//
//struct ContentView: View {
//    var body: some View {
//        NavigationView {
//            VStack {
//                // Assuming WebSocketView has all necessary initializations handled internally
//                WebSocketView()
//            }
//            .navigationTitle("ChineseCLIP")
//        }
//    }
//}
import SwiftUI

struct ContentView: View {
    @EnvironmentObject var viewModel: WebSocketViewModel  // Ensure it uses the same ViewModel

    var body: some View {
        NavigationView {
            VStack {
                WebSocketView()  // Already using the ViewModel from the environment
            }
            .navigationTitle("ChineseCLIP")
        }
    }
}
