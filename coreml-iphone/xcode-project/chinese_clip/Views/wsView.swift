//import SwiftUI
//import Combine
//
//class WebSocketViewModel: ObservableObject {
//    @Published var serverAddress = "ws://10.16.50.133:8765" // Default address
//    @Published var logMessages: [String] = []
//    @Published var isConnected = false
//
//    var imageClient: WebSocketImageClient?
//    var imgEncoder: ImgEncoder?
//
//    init() {
//        do {
//            let baseURL = URL(fileURLWithPath: Bundle.main.resourcePath!)
////            appendLog("ImgEncoder initialized with baseURL: \(baseURL)")
//            self.imgEncoder = try ImgEncoder(resourcesAt: baseURL)
//        } catch {
//            appendLog("Initialization of ImgEncoder failed")
//            self.imgEncoder = nil
//        }
//    }
//
//    func connect() {
//        guard let url = URL(string: serverAddress), let encoder = imgEncoder else {
//            appendLog("Invalid URL or encoder not initialized")
//            return
//        }
//        
//        imageClient = WebSocketImageClient(serverURL: url, encoder: encoder)
//        imageClient?.onConnectionStatusChanged = { [weak self] isConnected in
//            DispatchQueue.main.async {
//                self?.isConnected = isConnected
//            }
//        }
//        imageClient?.connect()
//        appendLog("Attempting to connect to \(serverAddress)")
//    }
//
//    func disconnect() {
//        imageClient?.disconnect()
//        appendLog("Disconnected")
//    }
//
//    private func appendLog(_ message: String) {
//        DispatchQueue.main.async {
//            self.logMessages.append(message)
//        }
//    }
//}
//
//
//struct WebSocketView: View {
//    @StateObject var viewModel = WebSocketViewModel() // Initialize your encoder appropriately
//
//    var body: some View {
//        VStack {
//            TextField("Server Address", text: $viewModel.serverAddress)
//                .textFieldStyle(RoundedBorderTextFieldStyle())
//                .padding()
//
//            Button(action: {
//                if viewModel.isConnected {
//                    viewModel.disconnect()
//                } else {
//                    viewModel.connect()
//                }
//            }) {
//                Text(viewModel.isConnected ? "Disconnect" : "Connect")
//                    .foregroundColor(.white)
//                    .padding()
//                    .background(viewModel.isConnected ? Color.red : Color.blue)
//                    .cornerRadius(10)
//            }
//
//            ScrollView {
//                ScrollViewReader { value in
//                    VStack(alignment: .leading) {
//                        ForEach(viewModel.logMessages, id: \.self) { msg in
//                            Text(msg)
//                                .frame(maxWidth: .infinity, alignment: .leading)
//                                .padding(3)
//                        }
//                        Rectangle() // Invisible marker
//                            .frame(width: 0, height: 0)
//                            .id("bottom")
//                    }
//                    .onAppear {
//                        value.scrollTo("bottom", anchor: .bottom)
//                    }
//                    .onChange(of: viewModel.logMessages) { oldValue, newValue in
//                        // Compare old and new values if needed
//                        if oldValue != newValue {
//                            value.scrollTo("bottom", anchor: .bottom)
//                        }
//                    }
//                    .frame(width: 350)
//                }
//            }
//            .frame(width: 350, height: 400)
//            .border(Color.gray, width: 1)
//            .cornerRadius(10) // Add rounded corners to ScrollView
//            .padding()
//        }
//        .padding()
//    }
//}
//
import SwiftUI
import Combine
import Foundation

class WebSocketViewModel: ObservableObject {
    @Published var serverAddress = "ws://10.16.50.133:8765" // Default address
    @Published var logMessages: [String] = []
    @Published var isConnected = false
    @Published var averageSpeed = "N/A"
    @Published var clientKey = "12345"
    

    var imageClient: WebSocketImageClient?
    var imgEncoder: ImgEncoder?
    
    private var lastProcessingTimes: [Date] = []

    init() {
        do {
            let baseURL = URL(fileURLWithPath: Bundle.main.resourcePath!)
            self.imgEncoder = try ImgEncoder(resourcesAt: baseURL)
            setupClient()
        } catch {
            appendLog("Initialization of ImgEncoder failed")
            self.imgEncoder = nil
        }
    }
    
    private func setupClient() {
        guard let encoder = imgEncoder else { return }
        let serverURL = URL(string: serverAddress)!
        imageClient = WebSocketImageClient(serverURL: serverURL, encoder: encoder)
        imageClient?.onConnectionStatusChanged = { [weak self] isConnected in
            DispatchQueue.main.async {
                self?.isConnected = isConnected
            }
        }
        imageClient?.onProcessingCompleted = { [weak self] in
            DispatchQueue.main.async {
                self?.recordProcessingTime()
            }
        }
    }

    func connect() {
        setupClient()  // 确保 client 是最新的
        imageClient?.connect(withKey: clientKey)
        appendLog("Attempting to connect to \(serverAddress) with key \(clientKey)")
    }

    func disconnect() {
        imageClient?.disconnect()
        appendLog("Disconnected")
    }

    private func appendLog(_ message: String) {
        DispatchQueue.main.async {
            self.logMessages.append(message)
        }
    }

    private func recordProcessingTime() {
        lastProcessingTimes.append(Date())
        lastProcessingTimes = lastProcessingTimes.filter { Date().timeIntervalSince($0) <= 5 }
        if lastProcessingTimes.count > 1 {
            let total = Double(lastProcessingTimes.count - 1)
            let duration = lastProcessingTimes.last!.timeIntervalSince(lastProcessingTimes.first!)
            let average = total / duration
            averageSpeed = String(format: "%.2f imgae/sec", average)
        } else {
            averageSpeed = "N/A"
        }
    }
}


struct WebSocketView: View {
    @EnvironmentObject var viewModel: WebSocketViewModel  // Use the environment object

    var body: some View {
        VStack {
            TextField("Server Address", text: $viewModel.serverAddress)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
            
            TextField("Key", text: $viewModel.clientKey)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()
            Button(action: {
                if viewModel.isConnected {
                    viewModel.disconnect()
                } else {
                    viewModel.connect()
                }
            }) {
                Text(viewModel.isConnected ? "Disconnect" : "Connect")
                    .foregroundColor(.white)
                    .padding()
                    .background(viewModel.isConnected ? Color.red : Color.blue)
                    .cornerRadius(10)
            }

            Text("Average Processing Speed: \(viewModel.averageSpeed)")
                .padding()

            ScrollView {
                ScrollViewReader { value in
                    VStack(alignment: .leading) {
                        ForEach(viewModel.logMessages, id: \.self) { msg in
                            Text(msg)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(3)
                        }
                        Rectangle() // Invisible marker
                            .frame(width: 0, height: 0)
                            .id("bottom")
                    }
                    .onAppear {
                        value.scrollTo("bottom", anchor: .bottom)
                    }
                    .onChange(of: viewModel.logMessages) { oldValue, newValue in
                        if oldValue != newValue {
                            value.scrollTo("bottom", anchor: .bottom)
                        }
                    }
                    .frame(width: 350)
                }
            }
            .frame(width: 350, height: 370)
            .border(Color.gray, width: 1)
            .cornerRadius(10) 
            .padding()
        }
        .padding()
    }
}
