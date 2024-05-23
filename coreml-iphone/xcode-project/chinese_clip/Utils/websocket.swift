//
//  websocket.swift
//  chinese_clip
//
//  Created by Yixuan Si on 5/23/24.
//

import Foundation
import Starscream
import UIKit
import CoreML

// Define custom errors for the WebSocket handling and image processing
enum WebSocketError: Error {
    case connectionError(String)
    case imageDataError
    case encodingError(String)
}

class WebSocketImageClient: WebSocketDelegate {
    private var socket: WebSocket?
    private var isConnected = false {
        didSet {
            onConnectionStatusChanged?(isConnected)
        }
    }
    private var imgEncoder: ImgEncoder
    var onConnectionStatusChanged: ((Bool) -> Void)?
    
    var onProcessingCompleted: (() -> Void)?  // 添加这个闭包来通知处理完成
    
    var clientKey: String?

    init(serverURL: URL, encoder: ImgEncoder) {
        var request = URLRequest(url: serverURL)
        request.timeoutInterval = 5
        self.imgEncoder = encoder

        socket = WebSocket(request: request)
        socket?.delegate = self
    }

    func connect(withKey key: String) {
//        isConnected = true
        socket?.connect()
        isConnected = true  // This should only be set in the .connected event
        // 延迟发送 key，直到连接真正建立
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            if self?.isConnected == true {
                self?.sendKey(key)
            }
        }
    }

    func disconnect() {
        socket?.disconnect()
        isConnected = false // This should trigger the didSet and call the closure
        print("Disconnected from WebSocket.")
    }

    // Implement the required method from WebSocketDelegate
    func didReceive(event: WebSocketEvent, client: WebSocketClient) {
        switch event {
        case .connected(let headers):
            isConnected = true
            print("WebSocket is connected: \(headers)")
        case .disconnected(let reason, let code):
            isConnected = false
            print("WebSocket is disconnected: \(reason) with code: \(code)")
        case .binary(let data):
            processImageData(data)
        case .error(let error):
            print("WebSocket encountered an error: \(String(describing: error))")
        default:
            break
        }
    }

    // Process and handle the image data received via WebSocket
    private func processImageData(_ data: Data) {
        guard let image = UIImage(data: data) else {
            print("Error: Data cannot be converted to UIImage")
            return
        }

        Task {
            do {
                let embedding = try await MLMultiArray(computeEmbedding(from: image))
                sendEmbedding(embedding)
                onProcessingCompleted?()
            } catch {
                print("Error processing image: \(error)")
            }
        }
    }

//    private func sendEmbedding(_ embedding: MLMultiArray) {
//        var resultString = "["
//        for i in 0..<embedding.count {
//            let value = embedding[i].floatValue
//            resultString += "\(value)"
//            if i < embedding.count - 1 {
//                resultString += ", "
//            }
//        }
//        resultString += "]"
//        socket?.write(string: resultString) {
//            print("Embedding data sent successfully.")
//        }
//    }
    private func sendKey(_ key: String) {
        guard isConnected else {
            print("Connection not established or key not set.")
            return
        }
        socket?.write(string: key) {
            print("Key '\(key)' sent successfully.")
        }
    }
    private func sendEmbedding(_ embedding: MLMultiArray) {
        let stringValues = (0..<embedding.count).map { embedding[$0].floatValue.description }
        let resultString = "[\(stringValues.joined(separator: ", "))]"
        socket?.write(string: resultString) {
            print("Embedding data sent successfully.")
        }
    }

    
    private func computeEmbedding(from image: UIImage) async throws -> MLShapedArray<Float32> {
        return try await imgEncoder.computeImgEmbedding(img: image)
    }
}
